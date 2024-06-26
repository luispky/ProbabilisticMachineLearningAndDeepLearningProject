import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

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
        num_samples = 1000

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

def save_plot_generated_samples(filename, samples, labels=None, path="../plots/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = path + filename + '.png'
    
    if labels is not None:
        mask = labels == 0
        plt.scatter(samples[:, 0][mask], samples[:, 1][mask], alpha=0.5, label='Normal')
        plt.scatter(samples[:, 0][~mask], samples[:, 1][~mask], alpha=0.5, label='Anomaly')
        plt.legend()
    else:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Generated Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(filename)

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

def plot_loss(losses, filename, path="../plots/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(path + filename + '.png')