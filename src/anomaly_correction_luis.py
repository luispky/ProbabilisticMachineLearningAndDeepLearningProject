import sys
import os

sys.path.append(os.path.abspath('..'))
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.utils import LinearNoiseScheduler, EMA
from src.utils import Probabilities, plot_categories, plot_loss
from src.modules import NoisePredictor
from src.denoising_diffusion_pm import DDPM


class AnomalyCorrection:
    def __init__(self):

        #!COMMON_PARAMS----------------------------------------------------------------------
        self.structure = None
        self.proba = Probabilities(self.structure)

        #!DDPM-------------------------------------------------------------------------------
        self.ddpm_scheduler = None
        self.ddpm_model = None
        self.diffusion = None

    def set_ddpm_scheduler(self, noise_time_steps=128,
                           dataset_shape=None):
        """
        Set the scheduler for the DDPM model.
        A different scheduler can be set for the training, sampling and inpainting.

        params:
        noise_time_steps: int, the number of noise timesteps
        dataset_shape: tuple or list, the shape of the dataset,
                        here mostly used for the feature dimension
        """
        self.ddpm_scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps,
                                                   dataset_shape=dataset_shape)

    def set_diffusion_model(self,
                            dataset_shape,
                            noise_time_steps=128,
                            time_dim_embedding=128,
                            concat_x_and_t=True,
                            feed_forward_kernel=True,
                            hidden_units=(64, 128, 64),
                            unet=False,
                            model_filename=None,
                            ):
        """
        Set the DDPM model for the diffusion process.
        Either set the model with the parameters or load it from a file.
        The scheduler must be set before setting the model.
        """
        ddpm_scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps,
                                                   dataset_shape=dataset_shape)

        if self.ddpm_model is None and model_filename is None:
            self.ddpm_model = NoisePredictor(time_dim=time_dim_embedding,
                                             dataset_shape=dataset_shape,
                                             concat_x_and_t=concat_x_and_t,
                                             feed_forward_kernel=feed_forward_kernel,
                                             hidden_units=hidden_units,
                                             unet=unet)
            self.diffusion = DDPM(ddpm_scheduler, self.ddpm_model)
        else:
            self.diffusion = DDPM(ddpm_scheduler)
            self.diffusion.load_model_pickle(model_filename)

    def train_diffusion_model(self, batch_size=16,
                              learning_rate=1e-3,
                              epochs=64,
                              beta_ema=0.999,
                              ddpm_loss_name='ddpm_loss',
                              model_name='noise_predictor_ema_model',
                              original_data_name='original_data'):
        """
        Train the diffusion model with the given parameters.
      
        Outputs:
        - The model is saved as a pickle file
        - Plots of the original data and the loss
        """
        assert self.diffusion is not None, 'The diffusion model must be set'
        x_indices = self.get_diffusion_dataset()

        # Plot the original distribution
        # Later we plot the generated samples to see compare if the distribution is similar
        plot_categories(x_indices, self.structure, original_data_name,
                        save_locally=True)

        x_indices_tensor = torch.tensor(x_indices, dtype=torch.float32)
        dataset = TensorDataset(x_indices_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ema = EMA(beta_ema)
        loss = self.diffusion.train(dataloader=dataloader,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    ema=ema)
        plot_loss(loss, ddpm_loss_name, save_locally=True)

        self.diffusion.save_model_pickle(self.diffusion.ema_model, model_name)

    def sample_diffusion_model(self, num_samples=1000,
                               sampled_data_name='sampled_data'):
        """
        Sample the diffusion model with the EMA model.

        Outputs:
        - The sampled data in indices space
        - Plot of the sampled data
        """
        assert self.diffusion is not None, 'The diffusion model must be set'
        sampled_logits = self.diffusion.sample(model=self.diffusion.model,
                                               samples=num_samples)
        x_indices_sampled = self.proba.logits_to_values(sampled_logits.cpu().numpy())
        plot_categories(x_indices_sampled, self.structure, sampled_data_name,
                        save_locally=True)
        return x_indices_sampled

    def inpainting_diffusion_model(self, x_indices_to_inpaint,
                                   masks,
                                   resampling_steps=10,
                                   x_indices_to_inpaint_name='data_to_inpaint'):
        """
        Modifies the features of the anomalous data to be more similar to the normal data.

        Inputs:
        - x_indices_to_inpaint: pandas DataFrame, the data to inpaint, all anomalies
        - masks: list or numpy array, the masks to apply to the data to inpaint
        - x_indices_to_inpaint_name: string, the name of the data to inpaint for plotting

        Outputs:
        - inpainted_data_values: list of numpy arrays, the inpainted data for each mask
        - Plot of the data to inpaint
        """

        assert self.diffusion is not None, 'The diffusion model must be set'
        assert isinstance(x_indices_to_inpaint, pd.DataFrame), 'The data to inpaint must be a pandas dataframe'
        assert isinstance(masks, (list, np.array)), 'The masks must be a list or a numpy array'

        # Distribution of the data to inpaint
        plot_categories(x_indices_to_inpaint, self.structure, x_indices_to_inpaint_name,
                        save_locally=True)

        # Convert the indices data to logits
        x_logits = torch.tensor(self.proba.values_to_logits(x_indices_to_inpaint.to_numpy()), dtype=torch.float32)

        # inpaint the data (in logit space) for as many masks as desired
        inpainted_data_values = []
        for mask in masks:
            # move mask to logits space
            mask = np.repeat(mask, self.structure, axis=1)
            # inpaint the data using the EMA model
            inpainted_instance_logits = self.diffusion.inpaint(model=self.diffusion.model,
                                                               original=x_logits,
                                                               mask=mask,
                                                               resampling_steps=resampling_steps)
            inpainted_data_values.append(self.proba.logits_to_values(inpainted_instance_logits))

        return inpainted_data_values
