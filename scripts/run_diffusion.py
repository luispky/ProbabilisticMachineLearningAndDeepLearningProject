import sys
import os
sys.path.append(os.path.abspath('..'))
import torch
import numpy as np
import argparse
import wandb
from src.utils import GaussianDataset, LinearNoiseScheduler, EMA, plot_generated_samples, plot_loss
from src.utils import plot_data_to_inpaint, SumCategoricalDataset, element_wise_label_values_comparison
from src.modules import NoisePredictor
from src.denoising_diffusion_pm import DDPM
from src.denoising_diffusion_pm import save_model_to_dir
from src.utils import Probabilities, plot_agreement_disagreement_transformation, plot_categories
from rich.console import Console
from rich.table import Table

def main_gaussian_data():

    # Are we going to use Weights and Biases?
    wandb_track = True
    
    #!ARGUMENTS AND HYPERPARAMETERS
    # define the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters that don't influence the model too much
    beta_ema = 0.999
    args.samples = 1000
    args.lr = 1e-3
    sampler_comment = 'ema_model'
    args.epochs = 64
    
    #  Hyperparameters that influence the model
    concat = False
    feed_forward_kernel = True
    hidden_units = [32, 64]
    noise_time_steps = 128  # 128 good value, try 256
    time_dim_embedding = 64  # >=32 works well
    experiment_number = '30'
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat}"

    loss_name = 'loss_' + experiment_number
    original_data_name = 'original_data_' + experiment_number
    sample_image_name = 'gen_samples_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_' + experiment_number
    inpainted_data_name = 'inpainted_data_' + experiment_number

    #!WEIGHTS AND BIASES 
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionModelGaussian", name=experiment_number)

        # Add all the hyperparameters to wandb
        wandb.config.update({"architecture": architecture_comment,
                            "feed_forward_kernel": feed_forward_kernel,
                            'noise_time_steps': noise_time_steps,
                            'time_dim_embedding': time_dim_embedding,
                            'epochs': args.epochs,
                            'scheduler': 'linear',
                            "sampler": sampler_comment, 
                            'samples': args.samples,
                            'learning_rate': args.lr,
                            'beta_ema': beta_ema,
        })
    
    #!DATASET TO TRAIN THE MODEL
    means = [[-4, -4], [8, 8], [-4, 7]]
    covariances = [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]
    num_samples_per_distribution = [1000, 2000, 1500]
    dataset_generator = GaussianDataset()
    dataset_generator.generate_dataset(means, covariances, num_samples_per_distribution)
    dataloader = dataset_generator.get_dataloader()
    dataset_shape = dataset_generator.get_dataset_shape()
    
    
    # define the components of the DDPM model: scheduler, model, EMA class
    ema = EMA(beta=beta_ema)
    
    
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    # scheduler = CosineNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(time_dim=time_dim_embedding, dataset_shape=dataset_shape,
                           concat_x_and_t=concat,
                           feed_forward_kernel=feed_forward_kernel, hidden_units=hidden_units)

    # Instantiate the DDPM modelcd 
    diffusion = DDPM(scheduler, model, args)

    # train the model
    train_losses = diffusion.train(dataloader, ema)

    # Plot the loss
    plot_loss(train_losses, loss_name)

    # generate samples 
    samples = diffusion.sample(diffusion.ema_model)

    # save the generated samples
    plot_generated_samples(samples, filename=sample_image_name)

    # generate inpainting samples
    noise_means = np.random.normal(0, 0.25, 2)
    noise_covariances = np.random.normal(0, 0.1, (2, 2))
    means = [[-4, -4], [8, 8], [-4, 7], [6, -4]]
    covariances = [[[2, 0], [0, 2]],
                   [[2, 0], [0, 2]],
                   [[2, 0], [0, 2]],
                   [[2, 0], [0, 2]]]
    means = [mean + noise_means for mean in means]
    covariances = [cov + noise_covariances for cov in covariances]
    num_samples_per_distribution = [1000, 2000, 1500, 2500]
    boolean_labels = [False, False, False, True]
    data_to_inpaint = dataset_generator.get_features_with_mask(means,
                                                               covariances,
                                                               num_samples_per_distribution,
                                                               boolean_labels)

    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']
    plot_data_to_inpaint(x, mask)

    # inpaint the masked data
    inpainted_data = diffusion.inpaint(diffusion.ema_model, x, mask)

    # save the inpainted data
    plot_generated_samples(inpainted_data, inpainted_data_name)

    wandb.finish()

def main_sum_categorical_data():
    
    # Are we going to use Weights and Biases?
    wandb_track = True
    
    #!ARGUMENTS AND HYPERPARAMETERS-----------------------------------------------------
    # define the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters that don't influence the model too much
    beta_ema = 0.999
    args.samples = 1000
    args.lr = 1e-3
    sampler_comment = 'ema_model'
    args.epochs = 64
    # save_model = False

    #  Hyperparameters that influence the model
    concat = True
    feed_forward_kernel = True
    # hidden_units = [128, 256,  512, 256, 128]
    # hidden_units = [64, 128, 64]
    hidden_units = [92*2]
    # concat = False
    # feed_forward_kernel = False
    # hidden_units = None
    unet = False
    noise_time_steps = 128  # 128 good value, try 256
    time_dim_embedding = 64  # >=32 works well
    experiment_number = '14'
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat}"
    # architecture_comment = f"2 encoders and decoders | concat(x, t): {concat}"
    
    loss_name = 'loss_' + experiment_number
    original_data_name = 'original_data_' + experiment_number
    sample_image_name = 'gen_samples_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_' + experiment_number
    inpainted_data_name = 'inpainted_data_' + experiment_number

    #!WEIGHTS AND BIASES----------------------------------------------------------------
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionModelSumCategorical", name=experiment_number)

        # Add all the hyperparameters to wandb
        wandb.config.update({"architecture": architecture_comment,
                            "feed_forward_kernel": feed_forward_kernel,
                            "unet": unet,
                            'noise_time_steps': noise_time_steps,
                            'time_dim_embedding': time_dim_embedding,
                            'epochs': args.epochs,
                            'scheduler': 'linear',
                            "sampler": sampler_comment, 
                            'samples': args.samples,
                            'learning_rate': args.lr,
                            'beta_ema': beta_ema,
        })
        
    #!DATASET TO TRAIN THE MODEL---------------------------------------------------------
    size = 3000
    threshold = 15
    structure = [2, 3, 5, 7, 11]
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    _ = dataset_generator.generate_dataset(remove_anomalies=True, logits=True)
    dataloader = dataset_generator.get_dataloader(with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape() 

    # Plot the original distribution
    # Later we plot the generated samples to see compare if the distribution is similar
    plot_categories(dataset_generator.label_values, structure, original_data_name)
    
    #!DDPM COMPONENTS: SCHEDULER, MODEL, EMA---------------------------------------------
    ema = EMA(beta=beta_ema)
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps,
                                     dataset_shape=dataset_shape)
    model = NoisePredictor(time_dim=time_dim_embedding,
                           dataset_shape=dataset_shape,
                           concat_x_and_t=concat,
                           feed_forward_kernel=feed_forward_kernel,
                           hidden_units=hidden_units,
                           unet=unet)
    
    # Probabilities class instance to process the data
    prob_instance = Probabilities(structure)
    
    #!DDPM: TRAINING AND SAMPLING-------------------------------------------------------
    # DDPM model
    diffusion = DDPM(scheduler, model, args)
    # DDPM training
    train_losses = diffusion.train(dataloader, ema)
    plot_loss(train_losses, loss_name, save_wandb=wandb_track)
    
    # DDPM sampling
    sampled_logits = diffusion.sample(diffusion.ema_model)[0]
    sampled_data = prob_instance.logits_to_values(sampled_logits.cpu().numpy())
    
    # Plot the sampled distribution
    plot_categories(sampled_data, structure, sample_image_name, save_wandb=wandb_track)
    
    # compute the kullback-leibler divergence
    # todo: I need to do the mapping back to the categorical values
    # from scipy.special import kl_div
    # P and Q from the frequencies of a pd.DataFrame with categorical values
    # kl_divergence = kl_div(P, Q).sum()
    
    #!INPAINTING------------------------------------------------------------------
    # generate inpainting samples
    size = 1500
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    data_to_inpaint = dataset_generator.get_features_with_mask(label_values_mask=True)
    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']

    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(diffusion.ema_model, x, mask)
    inpainted_data = prob_instance.logits_to_values(inpainted_data.cpu().numpy())
    
    # Distribution of the data to inpaint and the inpainted data
    plot_categories(dataset_generator.label_values, structure, data_to_inpaint_name)
    plot_categories(inpainted_data, structure, inpainted_data_name)
    
    #!METRICS--------------------------------------------------------------------------
    # Count the number of original anomalies
    y = data_to_inpaint['y'].numpy().squeeze()
    number_anomalies = np.sum(y)
    
    inpaint_detailed = element_wise_label_values_comparison(data_to_inpaint['label_values'], 
                                                      inpainted_data, 
                                                      data_to_inpaint['values_mask'])
    num_rows_differ, known_values, total_wrongly_changed_values = inpaint_detailed

    # Count the number of times the threshold is exceeded
    y_after = inpainted_data.sum(axis=1) > threshold
    number_remaining_anomalies = np.sum(y_after)
    percentage_change = (number_remaining_anomalies - number_anomalies) / number_anomalies * 100
    right_changes = (y & ~y_after).sum().item()
    wrong_changes = (~y & y_after).sum().item()
    
    # plot_agreement_disagreement_transformation(y, y_after, inpainted_data_name)
    
    #!SUMMARY--------------------------------------------------------------------------
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=40)
    table.add_column("Value")
    
    table.add_row("Anomalies before inpainting / Total:", f"{number_anomalies} / {size}")
    table.add_row("Remaining anomalies / Total:", f"{number_remaining_anomalies} / {size}")
    table.add_row("Percentage change (-100% desired):", f"{percentage_change:.2f}%")
    table.add_row("Correct changes (balance):", f"{right_changes} ({number_anomalies - right_changes})")
    table.add_row("Wrong changes (balance):", f"{wrong_changes} ({number_anomalies - wrong_changes})")
    table.add_row("Number of rows wrongly modified:", f"{num_rows_differ}({size})")
    table.add_row("Number of known values wrongly modified:", f"{total_wrongly_changed_values}({known_values})")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    if wandb_track:
        wandb.finish()  

if __name__ == '__main__':
    # main_gaussian_data()
    main_sum_categorical_data()
    
