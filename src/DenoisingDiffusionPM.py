import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# write get data such that it outputs a dataloader

class EMA:
    # Exponential Moving Average
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class LinearNoiseScheduler:
    r""""
    Class for the linear noise scheduler that is used in DDPM.
    """
    
    def __init__(self, num_timesteps, beta_start=1e-4, beta_end=2e-2):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
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
        \alpha_bar_{t} and \sqrt{1-\alpha_bar_{t}} are scalars
        We reshape them to match x_{0} and noise
        First we repeat the scalars to match the batch size
        Then we reshape them to match the original shape and perform broadcasting
        """
        batch_size = x0.shape[0]
        num_dims_to_add = len(x0.shape) - 1
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(x0.device)[t].reshape(batch_size).view(*([-1]+[1]*num_dims_to_add))
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(x0.device)[t].reshape(batch_size).view(*([-1]+[1]*num_dims_to_add))
        
        return sqrt_alpha_cum_prod * x0 + sqrt_one_minus_alpha_cum_prod * noise

class NoisePredictor(nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, x_t, t, y):
        pass
    
class Dataset:
    
    def __init__(self):
        self.dataset = None
        self.labels = None
        
        # pass

    def generate_data(self):
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
        dataloader = DataLoader(tensor_dataset, batch_size=14, shuffle=True)
        
        return dataloader    

    def plot_data(self):
        if self.dataset is None or self.labels is None:
            print('No dataset to plot')
            return
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
    def __init__(self, scheduler, model, args, dataset, ema):
        self.scheduler = scheduler
        self.model = model
        self.args = args
        self.dataset = dataset
        self.ema = ema
    
    def get_data(self):
        dataloader = self.dataset.generate_data()
        return dataloader

    def save_model(self):
        pass
    
    def train(self):
        # load the data
        dataloader = self.get_data()
        
        # copy the model and set it to evaluation mode
        ema_model = copy.deepcopy(self.model).eval().requires_grad_(False) 
        
        # use the AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # use the Mean Squared Error loss
        criterion = nn.MSELoss()
        
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

    def sample(self, n, labels, cfg_scale=3):
        
        pass

def main():
    # define the arguments
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 300
    args.batch_size = 14
    args.num_classes = 2
    args.device = "cpu"
    args.lr = 3e-4
    
    model = NoisePredictor()
    scheduler = LinearNoiseScheduler(num_timesteps=100)
    dataset = Dataset()
    ema = EMA(beta=0.995)
    
    diffusion_model = DDPM(scheduler, model, args, dataset, ema)
    
    diffusion_model.train()
    
    pass

def test():
    dataset = Dataset()
    dataset.generate_data()
    dataset.plot_data()
    
    # pass

if __name__ == '__main__':
    test()