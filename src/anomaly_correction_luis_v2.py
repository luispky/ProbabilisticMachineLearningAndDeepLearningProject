import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from src.utils import LinearNoiseScheduler, EMA
from src.utils import element_wise_label_values_comparison
from src.utils import Probabilities, plot_categories, plot_loss
from src.utils import CustomDataset
from src.modules import NoisePredictor
from src.denoising_diffusion_pm import DDPM
from rich.console import Console
from rich.table import Table


class AnomalyCorrection:
  def __init__(self, fuck_you_bro, # just kidding, lez do this again nano desu!
              # Reasonably common parameters
              dataframe_path = None,
              #  Omaru-sensei 🤓 hyperparameters
              bruh,
              # DDPM hyperparams
              ddpm_feed_forward_kernel=True, 
              ddpm_hidden_units=[64, 128, 64],
              ddpm_concat_x_and=True,
              ddpm_unet=False,
              ddpm_noise_time_steps=128,
              ddpm_time_dim_embedding=64,
              ddpm_epochs_ddpm=64,
              ddpm_synthetic_samples=1000,
              ddpm_lr=1e3,
              ddpm_beta_ema=0.999,
              ddpm_original_data_name = 'original_data',
              ddpm_sample_image_name = 'gen_samples',
              ddpm_data_to_inpaint_name = 'data_to_inpaint',
              ddpm_inpainted_data_name = 'inpainted_data', 
              ddpm_loss_name = 'ddpm_loss'
              ):
    
    #!COMMON_PARAMS----------------------------------------------------------------------
    self.dataframe_path = dataframe_path
    self.structure = None
    self.proba = Probabilities(self.structure)
    
    #!INVERSE_GRADIENT-------------------------------------------------------------------
    self.bruh = bruh
    self.probabiliy_thresholds = None
    
    #!DDPM-------------------------------------------------------------------------------
    
    # Values for the diffusion process: training, sampling and inpainting
    parser_ddpm = argparse.ArgumentParser()
    self.args_ddpm = parser_ddpm.parse_args()
    self.args_ddpm.lr = ddpm_lr
    self.args_ddpm.samples = ddpm_synthetic_samples
    self.args_ddpm.epochs = ddpm_epochs_ddpm
    self.args_ddpm.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # hyperparamters
    self.feed_forward_kernel = ddpm_feed_forward_kernel
    self.hidden_units = ddpm_hidden_units
    self.concat_x_and_t = ddpm_concat_x_and
    self.unet = ddpm_unet
    self.noise_time_steps = ddpm_noise_time_steps
    self.time_dim_embedding = ddpm_time_dim_embedding
    self.beta_ema = ddpm_beta_ema
    
    # variables to save the name of the plots if desired
    self.original_data_name = ddpm_original_data_name
    self.sampled_data_name = ddpm_sample_image_name
    self.data_to_inpaint_name = ddpm_data_to_inpaint_name
    self.inpainted_data_name = ddpm_inpainted_data_name
    
    """
    !Workaround for the dataset with overhead
    missing parameters/functions for the code
    which are otherwise easy to get with the proper classes 
    instead of putting it all here, np, I promise
    - DatasetClass
      - label_values
      - generate_dataset
      - get_dataloader
      - get_dataset_shape
    """


  def set_diffusion_parameters(self, 
                              ddpm_feed_forward_kernel=True, 
                              ddpm_hidden_units=[64, 128, 64],
                              ddpm_concat_x_and=True,
                              ddpm_unet=False,
                              ddpm_noise_time_steps=128,
                              ddpm_time_dim_embedding=64,
                              ddpm_epochs_ddpm=64,
                              ddpm_synthetic_samples=1000,
                              ddpm_lr=1e3,
                              ddpm_beta_ema=0.999,
                              ddpm_original_data_name = 'original_data',
                              ddpm_sample_image_name = 'gen_samples',
                              ddpm_data_to_inpaint_name = 'data_to_inpaint',
                              ddpm_inpainted_data_name = 'inpainted_data', 
                              ddpm_loss_name = 'ddpm_loss'):
    
    # Values for the diffusion process: training, sampling and inpainting
    parser_ddpm = argparse.ArgumentParser()
    self.args_ddpm = parser_ddpm.parse_args()
    self.args_ddpm.lr = ddpm_lr
    self.args_ddpm.samples = ddpm_synthetic_samples
    self.args_ddpm.epochs = ddpm_epochs_ddpm
    self.args_ddpm.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # hyperparamters
    self.feed_forward_kernel = ddpm_feed_forward_kernel
    self.hidden_units = ddpm_hidden_units
    self.concat_x_and_t = ddpm_concat_x_and
    self.unet = ddpm_unet
    self.noise_time_steps = ddpm_noise_time_steps
    self.time_dim_embedding = ddpm_time_dim_embedding
    self.beta_ema = ddpm_beta_ema
    
    # variables to save the name of the plots if desired
    self.original_data_name = ddpm_original_data_name
    self.sampled_data_name = ddpm_sample_image_name
    self.data_to_inpaint_name = ddpm_data_to_inpaint_name
    self.inpainted_data_name = ddpm_inpainted_data_name 
    self.ddpm_loss_name = ddpm_loss_name
    
  def set_diffusion_model(self, dataset_shape):
    scheduler = LinearNoiseScheduler(noise_timesteps=self.noise_time_steps,
                                     dataset_shape=dataset_shape)
    model = NoisePredictor(time_dim=self.time_dim_embedding,
                           dataset_shape=dataset_shape,
                           concat_x_and_t=self.concat_x_and_t,
                           feed_forward_kernel=self.feed_forward_kernel,
                           hidden_units=self.hidden_units,
                           unet=self.unet) 
    self.diffusion = DDPM(scheduler, model, self.args_ddpm)
     
  def train_diffusion_model(self, batch_size=16):
      x_indices = self.get_diffusion_dataset()
      x_indices_tensor = torch.tensor(x_indices, dtype=torch.float32)
      dataset =  TensorDataset(x_indices_tensor)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
      ema = EMA(self.beta_ema)
      loss = self.diffusion.train(dataloader, ema)
      plot_loss(loss, self.ddpm_loss_name, save_locally=True)
      
      
      
      


  def difussion_correction(self, data_to_inpaint, # dictionary with x and y
                                        # x: to inpaint, y: labels
                masks): # list of masks
    """Super Sequential Pipe waiting for the data_to_inpaint and masks"""
    
    #!WORKAROUND DATASET CLASS----------------------------------------------------------
    # Dataset for training the DDPM model
    dataset_generator = CustomDataset(self.dataframe_path)
    _ = dataset_generator.generate_dataset(remove_anomalies=True, logits=True)
    dataloader = dataset_generator.get_dataloader(with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape()
    
    # Plot the original distribution
    # Later we plot the generated samples to see compare if the distribution is similar
    plot_categories(dataset_generator.label_values, self.structure, self.original_data_name, 
                    save_locally=True)
    
    #!DDPM COMPONENTS: SCHEDULER, MODEL, EMA--------------------------------------------- 
    ema = EMA(self.beta_ema)
    scheduler = LinearNoiseScheduler(noise_timesteps=self.noise_time_steps,
                                     dataset_shape=dataset_shape)
    model = NoisePredictor(time_dim=self.time_dim_embedding,
                           dataset_shape=dataset_shape,
                           concat_x_and_t=self.concat_x_and_t,
                           feed_forward_kernel=self.feed_forward_kernel,
                           hidden_units=self.hidden_units,
                           unet=self.unet)
    
    #!DDPM: TRAINING AND SAMPLING-------------------------------------------------------
    # DDPM model
    diffusion = DDPM(scheduler, model, self.args_ddpm)
    # DDPM training
    loss = diffusion.train(dataloader, ema)
    plot_loss(loss, self.ddpm_loss_name, save_locally=True)
    
    # DDPM sampling
    sampled_logits = diffusion.sample(diffusion.ema_model)[0]
    sampled_data = self.proba.logits_to_values(sampled_logits.cpu().numpy())

    # Plot the sampled distribution
    plot_categories(sampled_data, self.structure, self.sampled_data_name, 
                    save_locally=True)
    
    #!INPAINTING------------------------------------------------------------------
    # Either read or generate the data to inpaint from the dataset_generator
    data_to_inpaint = None
    x, y = data_to_inpaint['x'], data_to_inpaint['y']
    masks = None # or masks
    
    # Convert the data (in probability space) to inpaint to logits
    x = ... # todo: clarify this
    
    # inpaint the data (in logit space) for as many masks as desired
    inpainted_data_values = []
    for mask in masks:
      # move mask to logits space
      mask = np.repeat(mask, self.structure, axis=1)
      inpainted_instance_logits = diffusion.inpaint(diffusion.ema_model,
                                             x,
                                             mask)
      #!yo, just a comment here. I am already convert the inpainted data to values
      #!you can do it your way if you want 
      inpainted_data_values.append(self.proba.logits_to_values(inpainted_instance_logits))

    # Distribution of the data to inpaint and the inpainted data
    # A picture is worth a thousand words
    # Take one instance of the inpainted data and plot it
    data_to_inpaint_values = self.proba.prob_to_values(x)
    plot_categories(data_to_inpaint_values, self.structure, self.data_to_inpaint_name, 
                    save_locally=True)
    plot_categories(inpainted_data_values[0], self.structure, self.inpainted_data_name, 
                    save_locally=True)
    
    # Take one instance of the inpainted data and convert it to the original dataframe representation
    corrected_dataframe = dataset_generator.categorical_encoder.indices_to_dataframe(inpainted_data_values[0])
    
    #Inpainting finished, yei!
    
    #!METRICS--------------------------------------------------------------------------
    # Count the number of original anomalies
    # Assumming y is a torch tensor of shape (batch_size, 1)
    y = y.numpy().squeeze()
    number_anomalies = np.sum(y)
    
    inpaint_detailed = element_wise_label_values_comparison(data_to_inpaint_values, 
                                                      inpainted_data_values, 
                                                      masks[0])
    num_rows_differ, known_values, total_wrongly_changed_values = inpaint_detailed
    size = len(y)
    
    # Compute the final class of the inpainted data
    y_after = self.classifier(torch.tensor(inpainted_data_values))
    y_after = y_after.numpy().squeeze()
    number_anomalies_after = np.sum(y_after)
    percentage_change = (number_anomalies_after - number_anomalies) / number_anomalies * 100
    right_changes = (y & ~y_after).sum().item()
    wrong_changes = (~y & y_after).sum().item()
    
    #!SUMMARY--------------------------------------------------------------------------
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=40)
    table.add_column("Value")
    
    table.add_row("Anomalies before inpainting / Total:", f"{number_anomalies} / {size}")
    table.add_row("Remaining anomalies / Total:", f"{number_anomalies_after} / {size}")
    table.add_row("Percentage change (-100% desired):", f"{percentage_change:.2f}%")
    table.add_row("Correct changes (balance):", f"{right_changes} ({number_anomalies - right_changes})")
    table.add_row("Wrong changes (balance):", f"{wrong_changes} ({number_anomalies - wrong_changes})")
    table.add_row("Number of rows wrongly modified:", f"{num_rows_differ}({size})")
    table.add_row("Number of known values wrongly modified:", f"{total_wrongly_changed_values}({known_values})")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    corrected_dataframe['target'] = pd.Series(y_after)
    
    return inpainted_data_values, corrected_dataframe
  
  def inverse_gradient(self):
    """Easy peasy lemon squeezy. Omar sensei 🤓 complicated"""
    pass
    
def main():
  pass 

if __name__ == '__main__':
  main()
    
    
    
    
  