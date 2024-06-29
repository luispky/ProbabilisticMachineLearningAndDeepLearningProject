import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import copy
from safetensors.torch import save_model as safe_save_model
from safetensors.torch import load_model as safe_load_model
import wandb

from .modules import NoisePredictor

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
        self.scheduler.send_to_device(self.args.device) 
        
    def load_model(self, model_params, filename, path="../models/"):
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
        
        # Initialize list to store losses during training
        train_losses = []        
        
        # run the training loop
        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            # verify if the dataloader has labels 
            has_labels = True if len(dataloader.dataset[0]) == 2 else False
            
            running_loss = 0.0
            num_elements = 0
            
            for i, batch_data in enumerate(dataloader):  # x_{0} ~ q(x_{0})
                optimizer.zero_grad()
                
                # extract data from the batch verifying if it has labels
                
                if has_labels:
                    batch_samples, labels = batch_data
                    labels = labels.to(self.args.device)
                else:
                    batch_samples = batch_data[0]
                    labels = None
                    
                batch_samples = batch_samples.to(self.args.device)
                
                # t ~ U(1, T)
                t = torch.randint(0, self.scheduler.noise_timesteps, (batch_samples.shape[0],)).to(self.args.device)
                # batch_samples.shape[0] is the batch size
                
                # noise = N(0, 1)
                noise = torch.randn_like(batch_samples).to(self.args.device)
                
                # x_{t-1} ~ q(x_{t-1}|x_{t}, x_{0})
                # the scheduler is different from the C-FG paper
                x_t = self.scheduler.add_noise(batch_samples, noise, t) 
                
                # Classifier Free Guidance Training
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
                
                running_loss += loss.item()
                num_elements += batch_samples.shape[0]
                
                if ema is not None:
                    # update the EMA model
                    ema.step_ema(self.ema_model, self.model)
            
            epoch_loss = running_loss / num_elements
            train_losses.append(epoch_loss)
            
            pbar.set_description(f'Epoch: {epoch+1} | Loss: {epoch_loss:.4f}')
                
            wandb.log({'loss': epoch_loss})
            
        print('Training Finished')
    
        return train_losses

    # Sampling method according to the DDPM paper
    def sample(self, model, labels=None, cfg_strength=3):
        # ?: maybe something must be changed for the SumCategoricalDataset
        model.eval()
        model.to(self.args.device)
        samples_shape = self.model.architecture.dataset_shape
        
        labels = labels.to(self.args.device) if labels is not None else None
        
        print('Sampling...')
        
        with torch.no_grad():
            # x_{T} ~ N(0, I)
            x = torch.randn((self.args.samples, *samples_shape[1:])).to(self.args.device)
            ones = torch.ones(self.args.samples)
            # for t = T, T-1, ..., 1 (-1 in Python)
            for i in tqdm(reversed(range(self.scheduler.noise_timesteps)), position=0):
                
                t = (ones * i).long().to(self.args.device)
                predicted_noise = model(x, t, labels)
                
                # Classifier Free Guidance Sampling
                if cfg_strength > 0 and labels is not None:
                    uncond_predicted_noise = model(x, t, None)
                    # interpolate between conditional and unconditional noise
                    # C-FG paper formula:
                    predicted_noise = (1 + cfg_strength) * predicted_noise - cfg_strength * uncond_predicted_noise
                    
                # x_{t-1} ~ p_{\theta}(x_{t-1}|x_{t})
                x = self.scheduler.sample_prev_step(x, predicted_noise, t)
        
        model.train()
        
        print('Sampling Finished')
        
        return x

    # Inpainting method according to the RePaint paper
    def inpaint(self, model, original, mask, U=10):
        # todo: review the inpainting method to properly implement it
        # the parameters U is not totally clear
        # ?: maybe something must be changed for the SumCategoricalDataset
        model.eval()
        model.to(self.args.device)
        
        original = original.to(self.args.device)
        mask = mask.to(self.args.device)
        
        print('Inpainting...')
        
        with torch.no_grad():
            # x_{T} ~ N(0, I)
            x_t = torch.randn_like(original).to(self.args.device)
            x_t_minus_one = torch.randn_like(x_t)
            ones = torch.ones(x_t.shape[0])
            
            # for t = T, T-1, ..., 1 (-1 in Python)
            for i in tqdm(reversed(range(self.scheduler.noise_timesteps)), position=0):
                
                for u in range(U):
                    t = (ones * i).long().to(self.args.device)
                    
                    # epsilon = N(0, I) if t > 1 else 0
                    forward_noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
                    
                    # differs from the algorithm in the paper but doesn't maatter because of stochasticity
                    x_known = self.scheduler.add_noise(original, forward_noise, t)
                    
                    predicted_noise = model(x_t, t)
                    x_unknown = self.scheduler.sample_prev_step(x_t, predicted_noise, t)
                    
                    x_t_minus_one = x_known * mask + x_unknown * (~mask)
                    
                    x_t = self.scheduler.sample_current_state_inpainting(x_t_minus_one, t) if (u < U and i > 0) else x_t

        print('Inpainting Finished')
        
        return x_t_minus_one.to('cpu')


def save_model_to_dir(model, filename, path = "../models/"):
    """
    Save the model using safetensors.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    filename = path + filename + '.safetensors'
    
    safe_save_model(model, filename + '.safetensors')
    
    print(f'Model saved in {filename}')