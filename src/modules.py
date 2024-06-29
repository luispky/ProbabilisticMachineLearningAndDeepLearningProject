"""
! Observations:
* I need to modify the Architecture class to solve the specific problem at hand
* Encode the time steps better depending on the layers of the model
"""
import torch
import torch.nn as nn


class Architecture(nn.Module):
    r"""
    Neural network architecture for the noise predictor in DDPM.
    """
    def __init__(self, time_dim=256, dataset_shape=None):
        super().__init__()
        self.dataset_shape = dataset_shape
        
        # Time embedding layer
        # It ensures the time encoding is compatible with the noised samples
        self.emb_layer = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, dataset_shape[1]),
        )
        
        # Maybe add an embedding layer for each linear layer
        # todo: parameters should be passed as arguments
        out1 = 32
        out2 = 64
        out3 = 32
        
        # # Sequential layers
        self.blocks = nn.Sequential(
            nn.Linear(dataset_shape[1], out1),
            nn.ReLU(),
            nn.Linear(out1, out2),
            nn.ReLU(),
            nn.Linear(out2, out3),
            nn.ReLU(),
            nn.Linear(out3, dataset_shape[1]),
        )

    def forward(self, x_t, t):
        """
        The goal is to predict the noise for the diffusion model
        The architecture input is: x_{t} and t, which could be summed
        Thus, they must have the same shape
        x_t has shape (batch_size, ...)
        for instance, (batch_size, columns) where columns is x.shape[1]
                  or  (batch_size, rows, columns) where rows and columns are x.shape[1] and x.shape[2]
        """
        # Time embedding
        emb = self.emb_layer(t)  # emb has shape (batch_size, dataset_shape[1])
        
        # Cases for broadcasting emb to match x_t
        if x_t.shape == emb.shape:
            pass
        elif len(self.dataset_shape) == 3:
            emb = emb.unsqueeze(-1).expand_as(x_t)  # emb has shape (batch_size, dataset_shape[1], dataset_shape[2])
            # emb = emb.unsqueezep[:, :, None].repeat(1, 1, x_t.shape[-1])
            
        else:
            raise NotImplementedError('The shape of x_t is not supported')
            # add more cases for different shapes of x_t
        
        # Application of transformation layers
        x = self.blocks(x_t + emb)
        
        return x

class NoisePredictor(nn.Module):
    r"""
    Neural network for the noise predictor in DDPM.
    """
    
    def __init__(self, dataset_shape=None, time_dim=256, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.architecture = Architecture(time_dim, dataset_shape)
        
        # Label embedding layer
        # It encode the labels into the time dimension
        # It is used to condition the model
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)  # label_emb(labels) has shape (batch_size, time_dim)
        
    def forward(self, x_t, t, y=None):
        
        # Positional encoding for the time steps
        t = t.unsqueeze(-1).type(torch.float)  # t has shape (batch_size,1)
        # Now, t has shape (batch_size)
        t = positional_encoding(t, self.time_dim) 
        # t has shape (batch_size, time_dim)
        
        # Label embedding
        if y is not None:
            # y has shape (batch_size, 1)
            y = y.squeeze(-1)
            # y now has shape (batch_size)
            if y.dtype != torch.long:
                y = y.long()
            t += self.label_emb(y)
            # label_emb(y) has shape (batch_size, time_dim)
            # the sum is element-wise
        
        # Apply the architecture 
        output = self.architecture(x_t, t)
        return output


def positional_encoding(time_steps, embedding_dim):
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
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
    ).to(time_steps.device)
    pos_enc_a = torch.sin(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
    pos_enc_b = torch.cos(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc