import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseArchitectureKernel(nn.Module, ABC):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @abstractmethod
    def forward(self, x):
        pass


class FeedForwardKernel(BaseArchitectureKernel):
    def __init__(self, input_dim, output_dim, hidden_units: list):
        super().__init__(input_dim, output_dim)
        
        layers = [] 
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        for i in range(len(hidden_units)-1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    
    
class UNet1ChannelKernel(nn.Module):
    def __init__(self, input_dim, output_dim, concat_x_and_t):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.concat_x_and_t = concat_x_and_t
        
        input_channels = 1 
        output_channels = 1
        
        self.encoder1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.final_conv = nn.Conv1d(64, output_channels, kernel_size=1)
        
        self.projection_layer = nn.Linear(input_dim, output_dim) if concat_x_and_t else nn.Identity()
            
    def forward(self, x):
        enc1 = self.encoder1(x.unsqueeze(1))
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        x = self.final_conv(dec1).squeeze(1)
        return self.projection_layer(x)
    
    
class NoisePredictor(nn.Module):
    """
    Neural network for the noise predictor in DDPM.
    """
    
    def __init__(self, dataset_shape=None,
                 time_dim_emb=128,
                 num_classes=None,
                 feed_forward_kernel=True,
                 hidden_units: list | None=None, 
                 concat_x_and_t=False,
                 unet=False):
        super().__init__()
        
        if feed_forward_kernel:
            assert hidden_units is not None, 'The hidden units must be provided'
        else:
            assert hidden_units is None, 'The hidden units must not be provided'
        assert dataset_shape is not None, 'The dataset shape must be provided'
        assert len(dataset_shape) in [2, 3], 'The dataset shape is not supported'
        
        self.time_dim_emb = time_dim_emb
        self.dataset_shape = dataset_shape
        self.concat_x_and_t = concat_x_and_t
        
        input_dim = dataset_shape[1]
        if concat_x_and_t:
            input_dim += time_dim_emb

        # Time embedding layer
        # It ensures the time encoding is compatible with the noised samples
        self.time_emb_layer = nn.Sequential(
            nn.Linear(time_dim_emb, time_dim_emb),
            nn.SiLU(),
            nn.Linear(time_dim_emb, dataset_shape[1]),
        ) if not concat_x_and_t else nn.Identity()
        
        if feed_forward_kernel:
            self.architecture_kernel = FeedForwardKernel(input_dim, dataset_shape[1], hidden_units)
        elif unet:
            self.architecture_kernel = UNet1ChannelKernel(input_dim, dataset_shape[1], concat_x_and_t)
        else:
            raise NotImplementedError('The kernel is not implemented')
        
        # Label embedding layer
        # It encode the labels into the time dimension
        # It is used to condition the model
        self.num_classes = num_classes
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim_emb)  # label_emb(labels) has shape (batch_size, time_dim_emb)
            
    def positional_encoding(self, time_steps):
        r"""
        Sinusoidal positional encoding for the time steps.
        The sinusoidal positional encoding is a way of encoding the
        position of elements in a sequence using sinusoidal functions.
        It is defined as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        where 
            - pos is the position of the element in the sequence, 
            - i is the dimension of the positional encoding, and 
            - d_model is the dimension of the input
        The sinusoidal positional encoding is used in the Transformer model
        to encode the position of elements in the input sequence.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.time_dim_emb, 2).float() / self.time_dim_emb)).to(time_steps.device)
        pos_enc = torch.cat([torch.sin(time_steps * inv_freq), torch.cos(time_steps * inv_freq)], dim=-1)
        return pos_enc

    def forward(self, x_t, t, y=None):
        """
        The goal is to predict the noise for the diffusion model
        The architecture input is: x_{t} and t, which could be summed
        Thus, they must have the same shape
        x_t has shape (batch_size, ...)
        for instance, (batch_size, columns) where columns is x.shape[1]
                  or  (batch_size, rows, columns) where rows and columns are x.shape[1] and x.shape[2]
        """
        
        # Positional encoding for the time steps
        # t.shape (batch_size,1) -> (batch_size) -> (batch_size, time_dim_emb)
        t = self.positional_encoding(t.unsqueeze(-1).float()) 
        
        # Label embedding
        if y is not None:
            assert self.num_classes is not None, 'The number of classes must be provided'
            # y has shape (batch_size, 1) 0 -> (batch_size)
            y = y.squeeze(-1).long()
            t += self.label_emb(y)
            # label_emb(y) has shape (batch_size, time_dim_emb)
            # the sum is element-wise
        
        # Time embedding
        # t has shape (batch_size, time_dim_emb)
        emb = self.time_emb_layer(t)  # emb has shape (batch_size, ...) if concat_x_and_t is False
        # emb is of datatype float = torch.float32
        
        # Cases for broadcasting emb to match x_t
        if x_t.dim() == emb.dim():
            pass
        else:
            for _ in range(x_t.dim() - 2):
                emb = emb.unsqueeze(-1).expand_as(x_t)  # emb has shape (batch_size, ...)
                # emb = emb.unsqueezep[:, :, None].repeat(1, 1, x_t.shape[-1])
        
        # Application of transformation layers
        # torch layers work better with float32
        # thus we convert x_t to float32
        x_t = x_t.float()
        x_t = torch.cat((x_t, emb), dim=1) if self.concat_x_and_t else x_t + emb
        
        return self.architecture_kernel(x_t)