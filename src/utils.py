import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from abc import ABC, abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader


class InpaintingData:
    """ Author: Luis
    Class to generate the dataset for the diffusion model, with masking for the 4th distribution.
    """
    
    def __init__(self):
        self.dataset = None
        self.mask = None

    def generate_data(self):
        # Define the number of samples to generate
        num_samples = 2000
        
        # Define means and covariances with added randomness
        mean1 = [-5, -5] + np.random.normal(0, 0.25, 2)
        cov1 = [[1.5, 0], [0, 1.5]] + np.random.normal(0, 0.1, (2, 2))
        
        mean2 = [9, 9] + np.random.normal(0, 0.25, 2)
        cov2 = [[1.5, 0], [0, 1.5]] + np.random.normal(0, 0.1, (2, 2))
        
        mean3 = [-5, 8] + np.random.normal(0, 0.25, 2)
        cov3 = [[1.5, 0], [0, 1.5]] + np.random.normal(0, 0.1, (2, 2))
        
        mean4 = [7, -5] + np.random.normal(0, 0.25, 2)
        cov4 = [[1.5, 0], [0, 1.5]] + np.random.normal(0, 0.1, (2, 2))

        # Ensure covariance matrices are positive semi-definite
        cov1 = np.dot(cov1, cov1.T)
        cov2 = np.dot(cov2, cov2.T)
        cov3 = np.dot(cov3, cov3.T)
        cov4 = np.dot(cov4, cov4.T)
        
        # Generate the samples
        samples1 = np.random.multivariate_normal(mean1, cov1, num_samples)
        samples2 = np.random.multivariate_normal(mean2, cov2, num_samples)
        samples3 = np.random.multivariate_normal(mean3, cov3, num_samples)
        samples4 = np.random.multivariate_normal(mean4, cov4, num_samples)

        # Concatenate the samples to create the dataset
        self.dataset = np.concatenate((samples1, samples2, samples3, samples4), axis=0)

        # Create mask: False for samples from distribution 4, True otherwise
        self.mask = np.ones(4 * num_samples, dtype=bool)
        self.mask[3 * num_samples:] = False
        
        self.mask = self.mask[:, np.newaxis]
        
        # Convert dataset and mask to torch tensors
        dataset_tensor = torch.tensor(self.dataset, dtype=torch.float32)
        mask_tensor = torch.tensor(self.mask, dtype=torch.bool)

        return dataset_tensor, mask_tensor


class Dataset:
    r"""" Author: Luis
    Class to generate the dataset for the DDPM model.

    todo: name 'Dataset' is already part of torch.utils.data, plz consider renaming to something else
    """
    
    def __init__(self):
        self.dataset = None
        self.labels = None
        self.dataloader = None

    def generate_data(self, with_labels=True):
        # Check if the dataset is already generated
        if self.dataset is not None:
            print('Data already generated')
            return self.dataloader
        
        # Define the number of samples to generate
        num_samples = 2000

        # Define the mean and covariance of the four gaussians
        mean1 = [-4, -4]
        cov1 = [[2, 0], [0, 2]]

        mean2 = [8, 8]
        cov2 = [[2, 0], [0, 2]]

        mean3 = [-4, 7]
        cov3 = [[2, 0], [0, 2]]

        mean4 = [6, -4]
        cov4 = [[2, 0], [0, 2]]

        # todo cov1, cov2, cov3, cov4 may be of the wrong type, plz check
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
    """ Author: Luis
    # todo comment missing, nani tf daz dis duu  (0 w  0) ??
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig = plt.figure()
    
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
    plt.savefig(path + filename + '.png')
    
    wandb.log({filename: wandb.Image(fig)})


def plot_data_to_inpaint(dataset, mask):
    """ Author: Luis
    # todo comment
    """
    # Convert tensors to numpy arrays for plotting
    dataset_np = dataset.numpy()
    mask_np = mask.numpy().squeeze()

    # Extract samples from each distribution based on the mask
    samples1 = dataset_np[~mask_np]
    samples4 = dataset_np[mask_np]

    # Scatter plot of the dataset
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(samples1[:, 0], samples1[:, 1], c='blue', label='Reference')
    plt.scatter(samples4[:, 0], samples4[:, 1], c='red', label='Masked')
    plt.title('Dataset with Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    wandb.log({'Dataset with Mask': wandb.Image(fig)})


class EMA:
    """
    Exponential Moving Average
    This is a way to impose a smoother training process
    The weights of the model do not change abruptly
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        core idea of EMA
        the weights are an interpolation between the old and new weights weighted by beta
        """
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
        """
        # todo this method is static
        """
        ema_model.load_state_dict(model.state_dict())

        
def plot_loss(losses, filename, path="../plots/"):
    """ Author: Luis
    todo comment
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(path + filename + '.png')
    
    wandb.log({filename: wandb.Image(fig)})


class BaseNoiseScheduler(ABC):
    """ Author: Luis
    # todo comment
    """
    def __init__(self, noise_timesteps, dataset_shape):
        self.noise_timesteps = noise_timesteps
        num_dims_to_add = len(dataset_shape) - 1 
        self.num_dims_to_add = num_dims_to_add
    
    @abstractmethod
    def _initialize_schedule(self):
        pass

    def _send_to_device(self, device):
        """
        todo: fix all unresolved attributes (alphas, betas, ...)
        """
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

    def sample_prev_step(self, x_t, predicted_noise, t):
        r""""
        Reverse sampling method for diffusion
        x_{t-1} ~ p_{\theta}(x_{t-1}|x_{t})
        """
        
        # noise = z ~ N(0, I) if t > 1 else 0
        backward_noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
        
        mean = x_t - (self.betas[t] * predicted_noise) / self.sqrt_one_minus_alpha_cum_prod[t]
        mean = mean / torch.sqrt(self.alphas[t])
        std = (1.0 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t]) * self.betas[t]
        
        # x_{t-1} = predicted_mean_reconstruction + fixed_std * noise
        return mean + std * backward_noise

    def sample_current_state_inpainting(self, x_t_minus_one, t):
        r""""
        Resampling method for inpainting
        """
        
        # noise = z ~ N(0, I)
        noise = torch.randn_like(x_t_minus_one)
        
        return x_t_minus_one * torch.sqrt(self.alphas[t-1]) + self.betas[t-1] * noise


class LinearNoiseScheduler(BaseNoiseScheduler):
    r"""" Author: Luis
    Class for the linear noise scheduler that is used in DDPM.
    The dimensions of the noise scheduler parameters are expanded to match the
    dimensions of the samples of the dataset. 
    This is required to make broadcasting operations between the noise and the samples.
    This change is only added to the betas attribute and is propagated to the other attributes.
    """
    def __init__(self, noise_timesteps, dataset_shape, beta_start=1e-4, beta_end=2e-2):
        super().__init__(noise_timesteps, dataset_shape)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self._initialize_schedule()

    def _initialize_schedule(self):
        linspace = torch.linspace(self.beta_start, self.beta_end, self.noise_timesteps)    # note: Omar split this line
        self.betas = linspace.view(*([-1] + [1]*self.num_dims_to_add))                     # because it was too long
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)


class CosineNoiseScheduler(BaseNoiseScheduler):
    """Author: Luis
    # todo Needs improvement with offset
    """
    def __init__(self, noise_timesteps, s=0.008, dataset_shape=None):
        super().__init__(noise_timesteps, dataset_shape)
        self.s = torch.tensor(s)
        self._initialize_schedule()

    def _initialize_schedule(self):
        t = torch.linspace(0, self.noise_timesteps, self.noise_timesteps)
        f = lambda t: torch.cos((t / self.noise_timesteps + self.s) / (1 + self.s) * torch.pi / 2) ** 2
        alphas_bar = f(t) / f(0)   # todo: do not assign a lambda function to a variable, use def instead
        
        self.alphas = torch.ones_like(alphas_bar)
        self.alphas[1:] = alphas_bar[1:] / alphas_bar[:-1]
        self.alphas[0] = alphas_bar[0]
        self.betas = torch.clip(1-self.alphas, 0, 0.999)
        
        self.betas = self.betas.view(*([-1] + [1] * self.num_dims_to_add))
        self.alphas = self.alphas.view(*([-1] + [1] * self.num_dims_to_add))
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)


class Probabilities:
    """ Author: Omar
    This class helps normalize probabilities for a set
    of features with different number of values
    """
    def __init__(self, n_values: list):
        self.n_values = n_values
        self.n = len(n_values)
        self.length = sum(n_values)
        self.mat = None
        self._set_mat()

    def _set_mat(self):
        """Create binary masks that divide the various features"""
        self.mat = np.zeros((self.length, self.length), dtype=np.float64)
        for i in range(self.n):
            start = sum(self.n_values[:i])
            for j in range(self.n_values[i]):
                self.mat[start:start+j+1, start:start+self.n_values[i]] = 1

    def normalize(self, p):
        """Cap at 0, then normalize the probabilities for each feature"""
        assert len(p.shape) == 2, f'{len(p.shape)} != 2'
        assert p.shape[1] == self.length, f'{p.shape[1]} != {self.length}'
        p = np.maximum(0, p)
        s = np.dot(p, self.mat)
        assert np.all(s > 0), f'Zero sum: {s}'
        return p / s

    def to_onehot(self, x):
        """Convert the original values to one-hot encoding"""
        assert len(x.shape) == 2, f'{len(x.shape)} != 2'
        assert x.shape[1] == self.n, f'{x.shape[1]} != {self.n}'
        # check that each value of x is less than the number of values for that feature
        assert np.all(np.max(x, axis=0) < self.n_values), f'Values out of range'
        # check that values are positive
        assert np.all(x >= 0), f'Negative values'

        x1 = np.zeros((x.shape[0], self.length), dtype=np.float64)
        start = 0
        for i in range(self.n):
            x1[np.arange(x.shape[0]), x[:, i] + start] = 1
            start += self.n_values[i]
        return x1

    def onehot_to_values(self, x):
        """Return the original values from the one-hot encoding"""
        assert len(x.shape) == 2, f'{len(x.shape)} != 2'
        assert x.shape[1] == self.length, f'{x.shape[1]} != {self.length}'
        x1 = np.zeros((x.shape[0], self.n), dtype=np.int64)
        start = 0
        for i in range(self.n):
            x1[:, i] = np.argmax(x[:, start:start+self.n_values[i]], axis=1)
            start += self.n_values[i]
        return x1

    def prob_to_onehot(self, p):
        """Convert the probabilities to one-hot encoding"""
        assert len(p.shape) == 2, f'{len(p.shape)} != 2'
        assert p.shape[1] == self.length, f'{p.shape[1]} != {self.length}'
        x = np.zeros((p.shape[0], self.n), dtype=np.int64)
        start = 0
        for i in range(self.n):
            x[:, i] = np.argmax(p[:, start:start+self.n_values[i]], axis=1)
            start += self.n_values[i]
        return self.to_onehot(x)


class bcolors:
    """ Author: Omar
    This class helps to print colored text in the terminal.
    To uce this class, call cprint(text, color)
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(text, color, end='\n'):
    """ Author: Omar
    Colorful print function

    Usage:
    cprint('You may fluff her tai', bcolors.OKGREEN)
    cprint('Warning: no sadge allowed', bcolors.WARNING)
    cprint('Failed to be sadge', bcolors.FAIL)
    """
    print(color + text + bcolors.ENDC, end=end)
