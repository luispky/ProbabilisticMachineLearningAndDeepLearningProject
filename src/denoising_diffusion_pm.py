import sys 
import os
import numpy as np
import pandas as pd
import copy
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from safetensors.torch import save_model as safe_save_model
from safetensors.torch import load_model as safe_load_model

from .modules import NoisePredictor
from .utils import LinearNoiseScheduler, EMA
from .utils import plot_categories, plot_loss


class DDPM:
    """
    Class for the Denoising Diffusion Probabilistic Model.
    It just implements methods but not the model itself.
    It implements the training and sampling methods for the model according to the DDPM paper.
    It also includes additional components to allow conditional sampling according to the labels.
    From the CFDG papers the changes are minimal. 
    """
    def __init__(self,
                dataset_shape, 
                noise_time_steps,
                ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = LinearNoiseScheduler(noise_time_steps=noise_time_steps,
                                              dataset_shape=dataset_shape)
        
        # send the scheduler attributes to the device
        self.scheduler.send_to_device(self.device) 
        self.dataset_shape = dataset_shape
        self.model = None        
        self.ema_model = None
        self.conditional_training = False

    def set_model(self, time_dim_emb=128,
                 num_classes=None,
                 feed_forward_kernel=True,
                 hidden_units: list | None=None, 
                 concat_x_and_t=False,
                 unet=False 
                 ):
        """Create a new noise predictor model with the provided parameters."""
        print('Creating a new model...')
        self.model = NoisePredictor(dataset_shape=self.dataset_shape,
                            time_dim_emb=time_dim_emb,
                            num_classes=num_classes,
                            feed_forward_kernel=feed_forward_kernel,
                            hidden_units=hidden_units,
                            concat_x_and_t=concat_x_and_t,
                            unet=unet)
        
    def train(self, dataloader, learning_rate=1e-3, epochs=64, beta_ema=0.999, wandb_track=False):
        # Instantiate the Exponential Moving Average (EMA) class
        ema = EMA(beta_ema)
        
        # load the data
        assert dataloader is not None, 'Dataloader not provided'
        assert self.model is not None, 'Model not provided'
        assert isinstance(self.model, NoisePredictor), 'Model must be an instance of NoisePredictor'
        
        if ema is not None:
            # copy the model and set it to evaluation mode
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)         
        
        # use the AdamW optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        # use the Mean Squared Error loss
        criterion = nn.MSELoss()
        
        # send the model to the device
        self.model.to(self.device)
        
        # set the model to training mode
        self.model.train()
        
        print('Training...')
        
        # Initialize list to store losses during training
        train_losses = []        
        
        # verify if the dataloader has labels
        self.conditional_training = True if len(dataloader.dataset[0]) == 2 else False
        
        # run the training loop
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            
            running_loss = 0.0
            num_elements = 0
            
            for i, batch_data in enumerate(dataloader):  # x_{0} ~ q(x_{0})
                optimizer.zero_grad()
                
                # extract data from the batch verifying if it has labels
                
                if self.conditional_training:
                    batch_samples, labels = batch_data
                    labels = labels.to(self.device)
                else:
                    batch_samples = batch_data[0]
                    labels = None
                    
                batch_samples = batch_samples.to(self.device)
                
                # t ~ U(1, T)
                t = torch.randint(0, self.scheduler.noise_time_steps, (batch_samples.shape[0],)).to(self.device)
                # batch_samples.shape[0] is the batch size
                
                # noise = N(0, 1)
                noise = torch.randn_like(batch_samples).to(self.device)
                
                # x_{t-1} ~ q(x_{t-1}|x_{t}, x_{0})
                # the scheduler is different from the C-FG paper
                x_t = self.scheduler.add_noise(batch_samples, noise, t)
                
                # Classifier Free Guidance Training
                # If the labels are not provided, use them with a probability of 0.1
                # This allows the conditional model to be trained with and without labels
                # This is a form of data augmentation and allows the model to be more robust
                if self.conditional_training:
                    if np.random.rand() < 0.1:
                        labels = None
                
                # denoising step
                # noise_{theta} = NN(x_{t}, t)
                # with x_{t} = \sqrt{\alpha_bar_{t}}x_{t} + \sqrt{1-\alpha_bar_{t}}*noise
                #      t used for positional encoding
                # and  labels for conditional model

                # x_t = x_t.float()
                # t = t.int()

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
            
            if wandb_track:
                wandb.log({'loss': epoch_loss})
            
        print('Training Finished\n')
    
        return train_losses

    @torch.no_grad()
    def sample(self, samples, with_labels=False, num_classes=None, cfg_strength=3):
        """Sampling method according to the DDPM paper."""
        
        assert self.model is not None, 'Model not provided'
        assert isinstance(self.model, NoisePredictor), 'Model must be an instance of NoisePredictor'
        
        self.model.eval()
        self.model.to(self.device)
        
        if self.conditional_training:
            assert with_labels and num_classes is not None, 'The number of classes in the labels must be specified'
            assert cfg_strength > 0, 'The strength of the Classifier-Free Guidance must be positive'
            labels = torch.randint(0, num_classes, (samples,)).to(self.device)
        else:
            if with_labels:
                print('Model was not trained with labels. Labels not sampled.')
            labels = None
            
        print('Sampling...')
        
        # x_{T} ~ N(0, I)
        x = torch.randn((samples, *self.dataset_shape[1:])).to(self.device)
        ones = torch.ones(samples)
        # for t = T, T-1, ..., 1 (-1 in Python)
        pbar = tqdm(reversed(range(self.scheduler.noise_time_steps)))
        for i in pbar:
            
            t = (ones * i).long().to(self.device)
            predicted_noise = self.model(x, t, labels)
            
            # Classifier-Free Guidance Sampling
            # The C-FG paper uses a conditional model to sample the noise
            if self.conditional_training:
                if labels is not None:
                    uncond_predicted_noise = self.model(x, t, None)
                    # interpolate between conditional and unconditional noise
                    # C-FG paper formula:
                    predicted_noise = (1 + cfg_strength) * predicted_noise - cfg_strength * uncond_predicted_noise
                
            # x_{t-1} ~ p_{\theta}(x_{t-1}|x_{t})
            x = self.scheduler.sample_prev_step(x, predicted_noise, t)
        
        self.model.train()
        
        print('Sampling Finished\n')
        
        if labels is not None:
            return [x, labels]
        return [x]

    @torch.no_grad()
    def inpaint(self, original, mask, resampling_steps=10):
        """Inpainting method according to the RePaint paper."""
        # ?: implement
        
        assert self.model is not None, 'Model not provided'
        assert isinstance(self.model, NoisePredictor), 'Model must be an instance of NoisePredictor'
        
        # !The Repaint paper uses an unconditionally trained model to inpaint the image
        assert not self.conditional_training, 'Model must be unconditionally trained'
        
        self.model.eval()
        self.model.to(self.device)
        
        original = original.to(self.device)
        mask = mask.to(self.device)
        
        print('Inpainting...')
        
        # x_{T} ~ N(0, I)
        x_t = torch.randn_like(original).to(self.device)
        x_t_minus_one = torch.randn_like(x_t)
        ones = torch.ones(x_t.shape[0])
        
        # for t = T, T-1, ..., 1 (-1 in Python)
        pbar = tqdm(reversed(range(self.scheduler.noise_time_steps)))
        for i in pbar:
            
            for u in range(resampling_steps):
                t = (ones * i).long().to(self.device)
                
                # epsilon = N(0, I) if t > 1 else 0
                forward_noise = torch.randn_like(x_t) if t[0] > 0 else torch.zeros_like(x_t)
                
                # differs from the algorithm in the paper but doesn't matter because of stochasticity
                x_known = self.scheduler.add_noise(original, forward_noise, t)

                predicted_noise = self.model(x_t, t)
                x_unknown = self.scheduler.sample_prev_step(x_t, predicted_noise, t)
                
                # The mask is the opposite of the paper, they changed their notation and was published like that
                x_t_minus_one = mask * x_unknown + x_known * (~mask)
                
                x_t = self.scheduler.sample_current_state_inpainting(x_t_minus_one, t) if (u < resampling_steps and i > 0) else x_t

        print('Inpainting Finished\n')
        
        self.model.train()
        
        return x_t_minus_one
    
    def load_model_safe_tensors(self, time_dim,
                                num_classes,
                                concat_x_and_t,
                                feed_forward_kernel,
                                hidden_units,
                                unet, 
                                filename, path="../models/"):
        """
        Load model parameters from a file using safetensors.
        """
        print(f'Loading model...')
        try:
            self.model = NoisePredictor(
                                    dataset_shape=self.dataset_shape,
                                    time_dim=time_dim,
                                    num_classes=num_classes,
                                    feed_forward_kernel=feed_forward_kernel, 
                                    hidden_units=hidden_units,
                                    concat_x_and_t=concat_x_and_t,
                                    unet=unet).to(self.device)
        
            filename = path + filename + '.safetensors'
            self.model = safe_load_model(self.model, filename)
        except FileNotFoundError:
            print('Model not found')
            self.model = None
    
    def load_model_pickle(self, filename, path="../models/"):
        """Load model parameters from a file using pickle."""
        print(f'Loading model...')
        try:
            filename = path + filename + '.pkl'
            model = torch.load(filename)
            assert isinstance(model, NoisePredictor), 'Model must be an instance of NoisePredictor'
            self.model = model
        except FileNotFoundError:
            print('Model not found')
            self.model = None
        
    def save_model_safetensors(self, filename, ema_model=True, path="../models/"):
        """
        Save the model using safetensors.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        filename = path + filename + '.safetensors'
        
        if ema_model and self.ema_model is not None:
            safe_save_model(self.ema_model, filename + '.safetensors')
        elif self.model is not None:
            safe_save_model(self.model, filename + '.safetensors')
        
        print(f'Model saved in {filename}')

    def save_model_pickle(self, filename, ema_model=True, path="../models/"):
        """Save the model using pickle."""
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + filename + '.pkl'
        if ema_model and self.ema_model is not None:
            torch.save(self.ema_model, filename)
        elif self.model is not None:
            torch.save(self.model, filename)


class DDPMAnomalyCorrection(DDPM):
    """
    Class for the Denoising Diffusion Probabilistic Model for Anomaly Correction.
    It particularizes the DDPM class for anomaly correction problem with end to end indices data.
    """
    def __init__(self, dataset_shape, noise_time_steps):
        super().__init__(dataset_shape, noise_time_steps)
    
    def train(self, dataset,
              batch_size=16, 
              learning_rate=1e-3,
              epochs=64,
              beta_ema=0.999,
              plot_data=False,
              proba=None,
              original_data_name='ddpm_original_data',
              wandb_track=False):
        
        assert proba is not None, 'The structure must be provided'
        assert isinstance(dataset, pd.DataFrame), 'The dataset must be a pandas DataFrame'
        x_indices = dataset.values
        
        if plot_data:
            plot_categories(x_indices, proba.structure, original_data_name, save_locally=plot_data) 
        x_logits_tensor = torch.tensor(proba.values_to_logits(x_indices), dtype=torch.float64)
        tensor_dataset =  TensorDataset(x_logits_tensor)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
        
        loss = super().train(dataloader=dataloader,
                             learning_rate=learning_rate,
                             epochs=epochs,
                             beta_ema=beta_ema,
                             wandb_track=wandb_track)
        
        return loss
        
    def sample(self, num_samples=1000, 
               plot_data=False,
               proba=None,
               sampled_data_name='ddpm_sampled_data',
               ):
        
        assert proba is not None, 'The probabilities object must be provided'
        
        sampled_logits = super().sample(samples=num_samples)[0]
        
        x_indices_sampled = proba.logits_to_values(sampled_logits.cpu().numpy())
            
        if plot_data:
            plot_categories(x_indices_sampled, proba.structure, sampled_data_name, save_locally=plot_data)
        
        return x_indices_sampled
    
    def inpaint(self, anomaly_indices,
                masks,
                proba=None,
                resampling_steps=10,
                ):
        
        assert proba is not None, 'The probabilities object must be provided'

        # Convert the indices data to logits
        x_logits = torch.tensor(proba.values_to_logits(anomaly_indices), dtype=torch.float64)

        inpainted_indices = []
        for mask in masks:
            mask = np.repeat(mask, np.array(proba.structure), axis=0)
            mask = torch.tensor(mask)
            x_inpainted_logits = super().inpaint(original=x_logits,
                                                 mask=mask,
                                                 resampling_steps=resampling_steps)
            inpainted_indices.append(proba.logits_to_values(x_inpainted_logits.cpu().numpy()))

        inpainted_indices = np.array(inpainted_indices).squeeze(1)

        return inpainted_indices
