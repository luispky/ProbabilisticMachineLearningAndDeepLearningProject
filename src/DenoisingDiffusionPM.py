import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

"""
! Observations:
* The dataset function only generates data for one sample of values
  Maybe, something that includes more samples would be better
* I need to figure out which feed forward model to use
* I need to understand were to include the time embedding in the forward pass
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
    """
    
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2, dataset_shape=None):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        # self.alphas = 1. - self.betas
        # self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        # self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        # self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
        num_dims_to_add = len(dataset_shape) - 1
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).view(*([1]*num_dims_to_add + [-1]))
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, x0, noise, t):
        r"""
        Forward method for diffusion
        x_{t} = \sqrt{\alpha_bar_{t}}x_{0} + \sqrt{1-\alpha_bar_{t}}\epsilon
        x_{0} has shape (batch_size, ...)
        noise has shape (batch_size, ...)
        \alpha_bar_{t} and \sqrt{1-\alpha_bar_{t}} are scalars or arrays of shape (batch_size,)
        We reshape them to match x_{0} and noise
        First we repeat the scalars to match the batch size
        Then we reshape them to match the original shape and perform broadcasting
        """
        # batch_size = x0.shape[0]
        # num_dims_to_add = len(x0.shape) - 1

        # sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(x0.device)[t].reshape(batch_size).view(*([-1]+[1]*num_dims_to_add))        
        # sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(x0.device)[t].view(*([-1]+[1]*num_dims_to_add))
        # sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(x0.device)[t].view(*([-1]+[1]*num_dims_to_add))
        
        return self.sqrt_alpha_cum_prod * x0 + self.sqrt_one_minus_alpha_cum_prod * noise
    
    # def sample_x_t_minus_one(self, x_t, noise_pred, t):
    #     r"""
    #     Use the noise prediction by model to get
    #     x_{t-1} using x_{t} and the noise predicted
    #     """

    #     mean = x_t - (self.betas.to(x_t.device)[t] * noise_pred) / self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]
    #     mean = mean / torch.sqrt(self.alphas[t])

    #     if t[0] > 0: # works for scalar or array
    #         noise = torch.randn_like(x_t)
    #     else:
    #         noise = torch.zeros_like(x_t)
        
    #     std = (1.0 - self.alpha_cum_prod.to(x_t.device)[t - 1])/(1.0 - self.alpha_cum_prod.to(x_t.device)[t])      
    #     std = std *  self.betas.to(x_t.device)[t]
        
    #     x_t_minus_one = mean + std * noise
        
    #     return x_t_minus_one

class Architecture(nn.Module):
    def __init__(self, time_dim=256, dataset_shape=None):
        super().__init__()
        self.dataset_shape = dataset_shape
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dataset_shape[1]),
        )
        
        # self.linear1 = nn.Linear(dataset_shape[1], dataset_shape[1])
        # self.linear2 = nn.Linear(dataset_shape[1], dataset_shape[1])
        # self.linear3 = nn.Linear(dataset_shape[1], dataset_shape[1])
        # self.relu = nn.ReLU()
        
        self.block = nn.Sequential(
            nn.Linear(dataset_shape[1], dataset_shape[1]),
            nn.ReLU(),
            nn.Linear(dataset_shape[1], dataset_shape[1]),
            nn.ReLU(),
            nn.Linear(dataset_shape[1], dataset_shape[1]),  
        )

    def forward(self, x_t, t, y):
        # x_t has shape (batch_size, ...)
        # for instance, (batch_size, columns) where columns is x.shape[1]
        #           or  (batch_size, rows, columns) where rows and columns are x.shape[1] and x.shape[2]

        # I have to sum x_t and t
        # t has shape (batch_size, time_dim)
        
        emb = self.emb_layer(t)
        # emb has shape (batch_size, dataset_shape[1])
        
        # Cases for broadcasting emb to match x_t
        if x_t.shape == emb.shape:
            pass
        elif len(self.dataset_shape) == 3:
            emb = emb.unsqueeze(-1).expand_as(x_t)
            # emb = emb.unsqueezep[:, :, None].repeat(1, 1, x_t.shape[-1])
            # emb has shape (batch_size, dataset_shape[1], dataset_shape[2])
            # where dataset_shape[2] = x_t.shape[2]
        else:
            raise NotImplementedError('The shape of x_t is not supported')
            # add more cases for different shapes of x_t
        
        x = self.block(x_t + emb)
        return x

class NoisePredictor(nn.Module):
    
    def __init__(self, time_dim=256, dataset_shape=None, num_classes=None):
        super().__init__()
        self.architecture = Architecture(time_dim, dataset_shape)
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
            # label_emb(labels) has shape (batch_size, time_dim)
        
    def positional_encoding(self, time_steps, embedding_dim):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, embedding_dim, 2, device=self.device).float() / embedding_dim)
        )
        pos_enc_a = torch.sin(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(time_steps.repeat(1, embedding_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
        
    def forward(self, x_t, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.positional_encoding(t, self.time_dim)
        # t has shape (batch_size, time_dim)
        
        if y is not None:
            t += self.label_emb(y)
            # label_emb(y) has shape (batch_size, time_dim)
            # the sum is element-wise

        output = self.architecture(x_t, t, y)
        return output

class Dataset:
    
    def __init__(self):
        self.dataset = None
        self.labels = None
        

    def generate_data(self):
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

        # Create labels for the samples
        labels1 = np.zeros((num_samples, 1)) # label 0 for samples1
        labels2 = np.zeros((num_samples, 1)) # label 0 for samples2
        labels3 = np.zeros((num_samples, 1)) # label 0 for samples3
        labels4 = np.ones((num_samples, 1))  # label 1 for samples4

        # Concatenate the samples to create the dataset
        self.dataset = np.concatenate((samples1, samples2, samples3, samples4), axis=0)

        # Concatenate the labels
        self.labels = np.concatenate((labels1, labels2, labels3, labels4), axis=0)
        
        # Transform the dataset and labels to torch tensors
        dataset = torch.tensor(self.dataset, dtype=torch.float32)
        labels = torch.tensor(self.labels, dtype=torch.float32)
        
        # Create a tensor dataset
        tensor_dataset = TensorDataset(dataset, labels)
        
        # Create a dataloader
        self.dataloader = DataLoader(tensor_dataset, batch_size=14, shuffle=True)
        
        return self.dataloader
    
    def get_dataset_shape(self):
        assert self.dataset is not None, 'Dataset not generated'
        return self.dataset.shape

    def plot_data(self):
        # Generate the dataset
        self.generate_data()
        # Plot the dataset with different colors for different labels
        mask = self.labels.flatten() == 0
        plt.scatter(self.dataset[:, 0][mask], self.dataset[:, 1][mask], alpha=0.5, label='Normal')
        plt.scatter(self.dataset[:, 0][~mask], self.dataset[:, 1][~mask], alpha=0.5, label='Anomaly')
        plt.title('2D Mixture of Gaussians')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

class DDPM:
    def __init__(self, scheduler, model, args, ema):
        self.scheduler = scheduler
        self.model = model
        self.args = args
        self.ema = ema
    
    def save_model(self):
        pass
    
    def train(self, dataloader=None):
        # load the data
        assert dataloader is not None, 'Dataloader not provided'
        
        # copy the model and set it to evaluation mode
        ema_model = copy.deepcopy(self.model).eval().requires_grad_(False) 
        
        # use the AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # use the Mean Squared Error loss
        criterion = nn.MSELoss()
        
        self.model.train()
        
        # run the training loop
        for epoch in range(self.args.epochs):
            losses = [] # change for tensorboard and include loggings
            
            pbar = tqdm(dataloader)
            for i, (batch_samples, labels) in enumerate(pbar): # x_{0} ~ q(x_{0})
                optimizer.zero_grad()
                
                batch_samples = batch_samples.to(self.args.device)
                labels = labels.to(self.args.device)
                
                # t ~ U(1, T)
                t = torch.randint(0, self.scheduler.num_timesteps, (batch_samples.shape[0],)).to(self.args.device)
                # batch_samples.shape[0] is the batch size
                
                # x_{t-1} ~ q(x_{t-1}|x_{t}, x_{0})
                # noise = N(0, 1)
                x_t, noise = self.scheduler.noise_images(batch_samples, t) 
                
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
                
                # 
                self.ema.step_ema(ema_model, self.model)

                losses.append(loss.item())
                # pbar.set_postfix(MSE=loss.item())
                pbar.set_description(f'Epoch: {epoch+1} | Loss: {loss.item()}')
            print(f'Finished epoch: {epoch+1} | Loss : {np.mean(losses)}')
            
        print('Training Finished')

    def sample(self, model, samples, labels, cfg_scale=3):
        model.eval()
        samples_shape = self.model.architecture.dataset_shape
        with torch.no_grad():
            # x_{T} ~ N(0, I)
            x = torch.randn((samples, *samples_shape[1:])).to(self.args.device)
            
            # for t = T, T-1, ..., 1
            for i in tqdm(reversed(range(1, self.scheduler.num_timesteps)), position=0):
                
                # This might give problems because here this is an array of shape (samples,)
                # instead of a simple scalar 
                t = (torch.ones(samples) * i).long().to(self.args.device)
                predicted_noise = model(x, t, labels)
                
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    # interpolate between conditional and unconditional noise
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                
                betas = self.scheduler.betas.to(x.device)
                sqrt_one_minus_alpha_cum_prod = self.scheduler.sqrt_one_minus_alpha_cum_prod.to(x.device)
                alphas = self.scheduler.alphas.to(x.device)
                alpha_cum_prod = self.scheduler.alpha_cum_prod.to(x.device)
                
                mean = x - (betas[t] * predicted_noise) / sqrt_one_minus_alpha_cum_prod[t]
                mean = mean / torch.sqrt(alphas[t])
                
                if t[0] > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                std = (1.0 - alpha_cum_prod[t - 1]) / (1.0 - alpha_cum_prod[t]) * betas[t]
                
                x = mean + std * noise  
        
        model.train()
                
        return x

def main():
    # define the arguments
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 300
    args.lr = 3e-4
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Dataset()
    dataloader = dataset.generate_data()
    dataset_shape = dataset.get_dataset_shape()
    model = NoisePredictor(dataset_shape = dataset_shape , num_classes=2)
    scheduler = LinearNoiseScheduler(num_timesteps=100, dataset_shape=dataset_shape)
    ema = EMA(beta=0.995)
    
    diffusion_model = DDPM(scheduler, model, args, ema)
    
    diffusion_model.train(dataloader)
    
    samples = 100
    diffusion_model.sample(ema, samples, )

def test():
    dataset = Dataset()
    dataset.plot_data()
    
if __name__ == '__main__':
    test()