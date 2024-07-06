import sys
import os
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import wandb
from src.utils import GaussianDataset, plot_generated_samples, plot_loss
from src.utils import plot_data_to_inpaint, SumCategoricalDataset, element_wise_label_values_comparison
from src.denoising_diffusion_pm import DDPM
from src.utils import Probabilities, plot_categories
from rich.console import Console
from rich.table import Table

# set default type to avoid problems with gradient
DEFAULT_TYPE = torch.float64
torch.set_default_dtype(DEFAULT_TYPE)

def main_gaussian_data():

    # Are we going to use Weights and Biases?
    wandb_track = False
    
    #!ARGUMENTS AND HYPERPARAMETERS
    # Hyperparameters that don't influence the model too much
    beta_ema = 0.999
    samples = 1000
    learning_rate = 1e-3
    epochs = 64
    inpaint_resampling = 20
    
    #  Hyperparameters that influence the model
    concat_x_and_t = True
    feed_forward_kernel = True
    # The Unet with one channel kernel doesn't work well with the gaussian data
    hidden_units = [32, 128, 32]
    noise_time_steps = 128  
    time_dim_emb = 64  # >=32 works well
    experiment_number = '01'
    model_name = f"diffusion_model_gaussian_{experiment_number}"
    # architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat_x_and_t}"
    architecture_comment = f"2 encoders and decoders | concat(x, t): {concat_x_and_t}"

    loss_name = 'loss_gaussian_' + experiment_number
    original_data_name = 'original_data_gaussian_' + experiment_number
    sample_image_name = 'gen_samples_gaussian_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_gaussian_' + experiment_number
    inpainted_data_name = 'inpainted_data_gaussian_' + experiment_number

    #!WEIGHTS AND BIASES 
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionModelGaussian", name=experiment_number)

        # Add all the hyperparameters to wandb
        wandb.config.update({"architecture": architecture_comment,
                            "feed_forward_kernel": feed_forward_kernel,
                            'noise_time_steps': noise_time_steps,
                            'time_dim_embedding': time_dim_emb,
                            'epochs': epochs,
                            'scheduler': 'linear',
                            'samples': samples,
                            'learning_rate': learning_rate,
                            'beta_ema': beta_ema,
                            'inpaint_resampling': inpaint_resampling,
        })
    
    #!DATASET TO TRAIN THE MODEL---------------------------------------------------------
    # Dataset parameters
    means = [[-4, -4], [8, 8], [-4, 7]]
    covariances = [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]
    num_samples_per_distribution = [1000, 2000, 1500]
    dataset_generator = GaussianDataset()
    dataset_generator.generate_dataset(means, covariances, num_samples_per_distribution)
    dataloader = dataset_generator.get_dataloader()
    dataset_shape = dataset_generator.get_dataset_shape()
    
    # Plot the original data
    dataset_generator.plot_data(original_data_name, save_locally=True)
    
    #!DDPM-------------------------------------------------------------------------------
    # Create a new instance
    diffusion = DDPM(dataset_shape=dataset_shape,
                     noise_time_steps=noise_time_steps,)
    
    # Load the model if it exists
    diffusion.load_model_pickle(model_name)
    
    if diffusion.model is None:
        diffusion.set_model(time_dim_emb=time_dim_emb,
                            concat_x_and_t=concat_x_and_t,
                            feed_forward_kernel=feed_forward_kernel,
                            hidden_units=hidden_units,
                            )  
        # DDPM training
        train_losses = diffusion.train(dataloader=dataloader,
                                       learning_rate=learning_rate, 
                                       epochs=epochs,
                                       beta_ema=beta_ema)
        plot_loss(train_losses, loss_name, save_wandb=wandb_track, save_locally=True)
        diffusion.save_model_pickle(filename=model_name, 
                                    ema_model=True)
    # DDPM sampling
    sampled_data = diffusion.sample(samples=samples)[0].cpu().numpy()
    
    # Plot the sampled distribution
    plot_generated_samples(sampled_data, filename=sample_image_name, save_locally=True)

    #!DATA TO INPAINT-------------------------------------------------------------------
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

    #!INPAINTING------------------------------------------------------------------


    # inpaint the masked data
    inpainted_data = diffusion.inpaint(x, mask, resampling_steps=inpaint_resampling).cpu().numpy()
    
    # Plots of the data to inpaint and the inpainted data
    plot_data_to_inpaint(x, mask, data_to_inpaint_name, save_locally=True)
    plot_generated_samples(inpainted_data, inpainted_data_name, save_locally=True)

    if wandb_track:
        wandb.finish()

def main_sum_categorical_data():
    
    # Are we going to use Weights and Biases?
    wandb_track = False
    
    #!ARGUMENTS AND HYPERPARAMETERS-----------------------------------------------------
    # Hyperparameters that don't influence the model too much
    beta_ema = 0.999
    samples = 1000
    learning_rate = 1e-3
    epochs = 64
    inpaint_resampling = 10
    
    #  Hyperparameters that influence the model
    concat_x_and_t = True
    feed_forward_kernel = True
    hidden_units = [92*2]
    unet = False
    noise_time_steps = 128  
    time_dim_emb = 64  
    experiment_number = '15'
    model_name = f"diffusion_model_categorical_{experiment_number}"
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat_x_and_t}"
    # architecture_comment = f"2 encoders and decoders | concat(x, t): {concat_x_and_t}"
    
    loss_name = 'loss_categorical_' + experiment_number
    original_data_name = 'original_data_categorical_' + experiment_number
    sample_image_name = 'gen_samples_categorical_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_categorical_' + experiment_number
    inpainted_data_name = 'inpainted_data_categorical_' + experiment_number

    #!WEIGHTS AND BIASES----------------------------------------------------------------
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionModelSumCategorical", name=experiment_number)

        # Add all the hyperparameters to wandb
        wandb.config.update({"architecture": architecture_comment,
                            "feed_forward_kernel": feed_forward_kernel,
                            "unet": unet,
                            'noise_time_steps': noise_time_steps,
                            'time_dim_embedding': time_dim_emb,
                            'epochs': epochs,
                            'scheduler': 'linear',
                            'samples': samples,
                            'learning_rate': learning_rate,
                            'beta_ema': beta_ema,
                            'inpaint_resampling': inpaint_resampling, 
        })
    
    # Dataset parameters
    threshold = 15
    structure = [2, 3, 5, 7, 11]
    # Probabilities class instance to process the data
    prob_instance = Probabilities(structure)

    #!DATASET TO TRAIN THE MODEL---------------------------------------------------------
    size = 3000
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    _ = dataset_generator.generate_dataset(remove_anomalies=True, logits=True)
    dataloader = dataset_generator.get_dataloader(with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape() 

    # Plot the original distribution
    # Later we plot the generated samples to see compare if the distribution is similar
    plot_categories(dataset_generator.label_values, structure, original_data_name, 
                    save_locally=True)
    
    #!DDPM-------------------------------------------------------------------------------
    # Create a new instance
    diffusion = DDPM(dataset_shape=dataset_shape,
                     noise_time_steps=noise_time_steps,)
    # Load the model if it exists
    diffusion.load_model_pickle(model_name)
    
    if diffusion.model is None:
        diffusion.set_model(time_dim_emb=time_dim_emb,
                            concat_x_and_t=concat_x_and_t,
                            feed_forward_kernel=feed_forward_kernel,
                            hidden_units=hidden_units,
                            unet=unet)  
        # DDPM training
        train_losses = diffusion.train(dataloader=dataloader,
                                       learning_rate=learning_rate, 
                                       epochs=epochs,
                                       beta_ema=beta_ema)
        plot_loss(train_losses, loss_name, save_wandb=wandb_track, save_locally=True)
        diffusion.save_model_pickle(filename=model_name, 
                                    ema_model=True)
    # DDPM sampling
    sampled_logits = diffusion.sample(samples=samples)[0]
    sampled_data = prob_instance.logits_to_values(sampled_logits.cpu().numpy())
    
    # Plot the sampled distribution
    plot_categories(sampled_data, structure, sample_image_name, save_wandb=wandb_track, 
                    save_locally=True)

    # todo: compute similarity between the original and the sampled distribution
    
    #!DATA TO INPAINT-------------------------------------------------------------------
    # generate inpainting samples
    size = 1500
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    data_to_inpaint = dataset_generator.get_features_with_mask(label_values_mask=True)
    dataset_shape = dataset_generator.get_dataset_shape()
    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']

    #!INPAINTING------------------------------------------------------------------
    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(original=x,
                                        mask=mask, 
                                        resampling_steps=inpaint_resampling)
    inpainted_data = prob_instance.logits_to_values(inpainted_data.cpu().numpy())
    
    # Distribution of the data to inpaint and the inpainted data
    plot_categories(dataset_generator.label_values, structure, data_to_inpaint_name, 
                    save_locally=True)
    plot_categories(inpainted_data, structure, inpainted_data_name, 
                        save_locally=True)
        
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
    
    table.add_row("Anomalies before inpainting / Total", f"{number_anomalies} / {size}")
    table.add_row("Remaining anomalies / Total", f"{number_remaining_anomalies} / {size}")
    table.add_row("Percentage change (-100% desired)", f"{percentage_change:.2f}%")
    table.add_row("Correct changes (balance)", f"{right_changes} ({number_anomalies - right_changes})")
    table.add_row("Wrong changes (balance)", f"{wrong_changes} ({number_anomalies - wrong_changes})")
    table.add_row("Number of rows wrongly modified", f"{num_rows_differ}({size})")
    table.add_row("Number of known values wrongly modified", f"{total_wrongly_changed_values}({known_values})")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    if wandb_track:
        wandb.finish()  

if __name__ == '__main__':
    main_gaussian_data()
    # main_sum_categorical_data()
    
