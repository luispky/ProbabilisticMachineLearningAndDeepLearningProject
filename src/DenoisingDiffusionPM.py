import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from safetensors.torch import save_model as safe_save_model
from safetensors.torch import load_model as safe_load_model
from safetensors.torch import save_file as safe_save_file
from safetensors.torch import load_file

"""
! Observations:
* I need to modify the Architecture class to solve the specific problem at hand
* I still need to implement inpainting and other methods
* I could modify the Noise Scheduler like in the OpenAI implementation
"""

class EMA:
    # Exponential Moving Average
    # This is a way to impose a smoother training process
    # The weights of the model do not change abruptly
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # core idea of EMA
        # the weights are an interpolation between the old and new weights weighted by beta
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        # warmup phase
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        # update the model average
        self.update_model_average(ema_model, model)
        self.step += 1
    
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class LinearNoiseScheduler:
    r""""
    Class for the linear noise scheduler that is used in DDPM.
    The dimensions of the noise scheduler parameters are expanded to match the
    dimensions of the samples of the dataset. 
    This is required to make broadcasting operations between the noise and the samples.
    This change is only added to the betas attribute and is propagated to the other attributes.
    """
    
    def __init__(self, noise_timesteps, beta_start=1e-4, beta_end=2e-2, dataset_shape=None):
        self.noise_timesteps = noise_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        num_dims_to_add = len(dataset_shape) - 1
        self.betas = torch.linspace(beta_start, beta_end, noise_timesteps).view(*( [-1] + [1]*num_dims_to_add ))
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
    def _send_to_device(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cum_prod = self.alpha_cum_prod.to(device)
        self.sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)
        self.sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)
    
    def add_noise(self, x0, noise, t):
        r"""
        Forward method for diffusion
        x_{t} = \sqrt{\alpha_bar_{t}}x_{0} + \sqrt{1-\alpha_bar_{t}}\epsilon
        x_{0} has shape (batch_size, ...)
        noise has shape (batch_size, ...)
        t has shape (batch_size,)
        The scheduler parameters already have the correct shape to match x_{0} and noise.
        """
        return self.sqrt_alpha_cum_prod[t] * x0 + self.sqrt_one_minus_alpha_cum_prod[t] * noise

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
            nn.SiLU(),
            nn.Linear(time_dim, dataset_shape[1]),
        )
        
        # Sequence of linear layers
        self.block = nn.Sequential(
            nn.Linear(dataset_shape[1], dataset_shape[1]),
            nn.ReLU(),
            nn.Linear(dataset_shape[1], dataset_shape[1]),
            nn.ReLU(),
            nn.Linear(dataset_shape[1], dataset_shape[1]),  
        )

    def forward(self, x_t, t):
        # The goal is to predict the noise for the diffusion model
        # The architecture input is: x_{t} and t, which could be summed
        # Thus, they must have the same shape
        # x_t has shape (batch_size, ...)
        # for instance, (batch_size, columns) where columns is x.shape[1]
        #           or  (batch_size, rows, columns) where rows and columns are x.shape[1] and x.shape[2]
        
        # Time embedding
        emb = self.emb_layer(t) # emb has shape (batch_size, dataset_shape[1])
        
        
        # Cases for broadcasting emb to match x_t
        if x_t.shape == emb.shape:
            pass
        elif len(self.dataset_shape) == 3:
            emb = emb.unsqueeze(-1).expand_as(x_t) # emb has shape (batch_size, dataset_shape[1], dataset_shape[2])
            # emb = emb.unsqueezep[:, :, None].repeat(1, 1, x_t.shape[-1])
            
        else:
            raise NotImplementedError('The shape of x_t is not supported')
            # add more cases for different shapes of x_t
        
        # Application of transformation layers
        x = self.block(x_t + emb)
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
            self.label_emb = nn.Embedding(num_classes, time_dim) # label_emb(labels) has shape (batch_size, time_dim)
        
    def positional_encoding(self, time_steps, embedding_dim):
        r"""
        Sinusoidal positional encoding for the time steps.
        The sinusoidal positional encoding is a way of encoding the position of elements in a sequence using sinusoidal functions.
        It is defined as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        where 
            - pos is the position of the element in the sequence, 
            - i is the dimension of the positional encoding, and 
            - d_model is the dimension of the input
        The sinusoidal positional encoding is used in the Transformer model to encode the position of elements in the input sequence.
        """
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        ).to(time_steps.device)
        pos_enc_a = torch.sin(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
        
    def forward(self, x_t, t, y=None):
        
        # Positional encoding for the time steps
        t = t.unsqueeze(-1).type(torch.float) # t has shape (batch_size,1)
        # Now, t has shape (batch_size)
        t = self.positional_encoding(t, self.time_dim) 
        # t has shape (batch_size, time_dim)
        
        # Label embedding
        if y is not None:
            # y has shape (batch_size, 1)
            y = y.squeeze(-1)
            # y now has shape (batch_size)
            if y.dtype!= torch.long:
                y = y.long()
            t += self.label_emb(y)
            # label_emb(y) has shape (batch_size, time_dim)
            # the sum is element-wise
        
        # Apply the architecture 
        output = self.architecture(x_t, t)
        return output
    
class Dataset:
    r""""
    Class to generate the dataset for the DDPM model.
    """
    
    def __init__(self):
        self.dataset = None
        self.labels = None

    def generate_data(self, with_labels=True):
        # Check if the dataset is already generated
        if self.dataset is not None:
            print('Data already generated')
            return self.dataloader
        
        # Define the number of samples to generate
        num_samples = 300

        # Define the mean and covariance of the four gaussians
        mean1 = [-4, -4]
        cov1 = [[1, 0], [0, 1]]

        mean2 = [8, 8]
        cov2 = [[2, 0], [0, 2]]

        mean3 = [-4, 7]
        cov3 = [[2, 0], [0, 2]]

        mean4 = [6, -4]
        cov4 = [[2, 0], [0, 2]]
        
        # Generate the samples
        samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)
        samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)
        samples3 = np.random.multivariate_normal(mean3, cov3, num_samples)
        samples4 = np.random.multivariate_normal(mean4, cov4, num_samples)

        # Concatenate the samples to create the dataset
        self.dataset = np.concatenate((samples1, samples2, samples3, samples4), axis=0)

        if with_labels:
            # Create labels for the samples
            labels1 = np.zeros((num_samples, 1)) # label 0 for samples1
            labels2 = np.zeros((num_samples, 1)) # label 0 for samples2
            labels3 = np.zeros((num_samples, 1)) # label 0 for samples3
            labels4 = np.ones((num_samples, 1))  # label 1 for samples4

            # Concatenate the labels
            self.labels = np.concatenate((labels1, labels2, labels3, labels4), axis=0)
            # labels.shape = (4*num_samples, 1)
        
        # Transform the dataset and labels to torch tensors
        dataset = torch.tensor(self.dataset, dtype=torch.float32)
        
        if with_labels:
            labels = torch.tensor(self.labels, dtype=torch.float32)
            
            # Create a tensor dataset
            tensor_dataset = TensorDataset(dataset, labels)
            
        else: 
            tensor_dataset = TensorDataset(dataset)
        
        # Create a dataloader
        self.dataloader = DataLoader(tensor_dataset, batch_size=14, shuffle=True)
        
        return self.dataloader
    
    def get_dataset_shape(self):
        assert self.dataset is not None, 'Dataset not generated'
        return self.dataset.shape

    def plot_data(self):
        # Generate the dataset
        self.generate_data(with_labels=True)
        # Plot the dataset with different colors for different labels
        mask = self.labels.flatten() == 0
        # labels.flatten() has shape (4*num_samples,)
        plt.scatter(self.dataset[:, 0][mask], self.dataset[:, 1][mask], alpha=0.5, label='Normal')
        plt.scatter(self.dataset[:, 0][~mask], self.dataset[:, 1][~mask], alpha=0.5, label='Anomaly')
        plt.title('2D Mixture of Gaussians')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

class DDPM:
    r"""
    Class for the Denoising Diffusion Probabilistic Model.
    It just implements methods but not the model itself.
    It implements the training and sampling methods for the model according to the DDPM paper.
    It also includes additional components to allow conditional sampling according to the labels.
    From the CFDG papers the changes are minimal. 
    """
    def __init__(self, scheduler, model, args):
        self.scheduler = scheduler
        self.model = model
        self.args = args
        self.ema_model = None

        # send the scheduler attributes to the device
        self.scheduler._send_to_device(self.args.device)
        
    def save_model(self, model, filename, path = "../models/"):
        if not os.path.exists(path):
            os.makedirs(path)
        
        filename = path + filename + '.safetensors'
        
        safe_save_model(model, filename + '.safetensors')
        
        print(f'Model saved in {filename}')
    
    def load_model(self, model_params, filename, path = "../models/"):
        r"""
        Load model parameters from a file using safetensors.
        """
        dataset_shape = model_params['dataset_shape']
        time_dim = model_params['time_dim']
        num_classes = model_params['num_classes']
        model = NoisePredictor(dataset_shape=dataset_shape, 
                               time_dim=time_dim, 
                               num_classes=num_classes).to(self.args.device)
        
        print(f'Loading model...')
        
        filename = path + filename + '.safetensors'
        return safe_load_model(model, filename)
    
    # Training method according to the DDPM paper
    def train(self, dataloader, ema=None):
        # load the data
        assert dataloader is not None, 'Dataloader not provided'
        
        if ema is not None:
            # copy the model and set it to evaluation mode
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.args.device)         
        
        # use the AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # use the Mean Squared Error loss
        criterion = nn.MSELoss()
        
        # send the model to the device
        self.model.to(self.args.device)
        
        # set the model to training mode
        self.model.train()
        
        print('Training...')
        
        # run the training loop
        for epoch in range(self.args.epochs):
            losses = [] # change for tensorboard and include loggings
            
            # verify if the dataloader has labels 
            has_labels = True if len(dataloader.dataset[0]) == 2 else False
            
            pbar = tqdm(dataloader)
            for i, batch_data in enumerate(pbar): # x_{0} ~ q(x_{0})
                optimizer.zero_grad()
                
                # extract data from the batch verifying if it has labels
                
                if has_labels:
                    batch_samples, labels = batch_data
                    labels = labels.to(self.args.device)
                else:
                    batch_samples = batch_data
                    labels = None
                    
                batch_samples = batch_samples.to(self.args.device)
                
                # t ~ U(1, T)
                t = torch.randint(0, self.scheduler.noise_timesteps, (batch_samples.shape[0],)).to(self.args.device)
                # batch_samples.shape[0] is the batch size
                
                # noise = N(0, 1)
                noise = torch.randn_like(batch_samples).to(self.args.device)
                
                # x_{t-1} ~ q(x_{t-1}|x_{t}, x_{0})
                x_t = self.scheduler.add_noise(batch_samples, noise, t) 
                
                # If the labels are not provided, use them with a probability of 0.1
                # This allows the conditional model to be trained with and without labels
                # This is a form of data augmentation and allows the model to be more robust
                if np.random.rand() < 0.1:
                    labels = None
                
                # denoising step
                # noise_{theta} = NN(x_{t}, t)
                # with x_{t} = \sqrt{\alpha_bar_{t}}x_{t} + \sqrt{1-\alpha_bar_{t}}*noise
                #      t used for positional encoding
                # and  labels for conditional model
                predicted_noise = self.model(x_t, t, labels)
                
                # compare the noise and predicted noise with loss metric
                loss = criterion(noise, predicted_noise)
                loss.backward()
                optimizer.step()
                
                if ema is not None:
                    # update the EMA model
                    ema.step_ema(self.ema_model, self.model)

                losses.append(loss.item())
                # pbar.set_postfix(MSE=loss.item())
                pbar.set_description(f'Epoch: {epoch+1} | Loss: {loss.item()}')
            print(f'Finished epoch: {epoch+1} | Loss : {np.mean(losses)}')
            
        print('Training Finished')

    # Sampling method according to the DDPM paper
    def sample(self, model, labels, cfg_scale=3):
        model.eval()
        model.to(self.args.device)
        samples_shape = self.model.architecture.dataset_shape
        
        print('Sampling...')
        
        with torch.no_grad():
            # x_{T} ~ N(0, I)
            x = torch.randn((self.args.samples, *samples_shape[1:])).to(self.args.device)
            
            # for t = T, T-1, ..., 1
            for i in tqdm(reversed(range(1, self.scheduler.noise_timesteps)), position=0):
                
                # This might give problems because here this is an array of shape (samples,)
                # instead of a simple scalar 
                t = (torch.ones(self.args.samples) * i).long().to(self.args.device)
                predicted_noise = model(x, t, labels)
                
                # Classifier Free Guidance Scale
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    # interpolate between conditional and unconditional noise
                    # lerp(x, y, alpha) = x * (1 - alpha) + y * alpha
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                
                # noise = z ~ N(0, I) if t > 1 else 0
                if t[0] > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # x_{t-1} ~ p_{\theta}(x_{t-1}|x_{t})
                betas = self.scheduler.betas
                sqrt_one_minus_alpha_cum_prod = self.scheduler.sqrt_one_minus_alpha_cum_prod
                alphas = self.scheduler.alphas
                alpha_cum_prod = self.scheduler.alpha_cum_prod
                
                # mean = x_{t} - const * predicted_noise 
                mean = x - (betas[t] * predicted_noise) / sqrt_one_minus_alpha_cum_prod[t]
                mean = mean / torch.sqrt(alphas[t])
                std = (1.0 - alpha_cum_prod[t - 1]) / (1.0 - alpha_cum_prod[t]) * betas[t]
                
                # x_{t-1} = predicted_mean_reconstruction + fixed_std * noise
                x = mean + std * noise  
        
        model.train()
        
        print('Sampling Finished')
        
        return x

def save_plot_generated_samples(filename, samples, labels=None, path="../plots/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = path + filename + '.png'
    
    if labels is not None:
        mask = labels == 0
        plt.scatter(samples[:, 0][mask], samples[:, 1][mask], alpha=0.5, label='Normal')
        plt.scatter(samples[:, 0][~mask], samples[:, 1][~mask], alpha=0.5, label='Anomaly')
    else:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Generated Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(filename)

def main():
    # define the arguments
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 300
    args.lr = 3e-4
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.samples = 1000
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    dataset = Dataset()
    dataloader = dataset.generate_data(with_labels=False)
    dataset_shape = dataset.get_dataset_shape()
    noise_time_steps = 1000
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    time_dim_embedding = 256
    model = NoisePredictor(dataset_shape = dataset_shape, time_dim=time_dim_embedding, num_classes=2)
    ema = EMA(beta=0.995)
    
    # Instantiate the DDPM model
    diffusion = DDPM(scheduler, model, args)
    
    # train the model
    diffusion.train(dataloader, ema)
    
    # save model
    # diffusion.save_model(diffusion.model, 'ddpm_model_03')
    
    labels = torch.randint(0, 2, (args.samples,)).to(args.device)
    samples = diffusion.sample(diffusion.ema_model, labels)
    
    # bring labels and samples to the cpu
    labels = labels.cpu().numpy()
    samples = samples.cpu().numpy()
    
    # save the generated samples
    save_plot_generated_samples('generated_samples_05', samples, labels) 

def test():
    dataset = Dataset()
    dataset.plot_data()
    
if __name__ == '__main__':
    main()