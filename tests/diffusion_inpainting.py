import sys
import os
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import wandb
from src.utils import GaussianDataset, plot_generated_samples, plot_loss
from src.utils import plot_data_to_inpaint, SumCategoricalDataset, element_wise_label_values_comparison
from src.denoising_diffusion_pm import DDPM
from src.utils import plot_categories, ClassificationModel
from datasets.generate_data import compute_divergence
from rich.console import Console
from rich.table import Table

# set default type to avoid problems with gradient
DEFAULT_TYPE = torch.float64
torch.set_default_dtype(DEFAULT_TYPE)
torch.set_default_device('cpu')

def main_gaussian_data():

    # Are we going to use Weights and Biases?
    wandb_track = True
    save_plots_locally = False
    
    #!ARGUMENTS AND HYPERPARAMETERS-----------------------------------------------------
    # Hyperparameters that don't influence the model too much
    beta_ema = 0.95
    samples = 1000
    learning_rate = 1e-3
    epochs = 128 # >= 64 doesn't improve much the model
    inpaint_resampling = 20
    
    #  Hyperparameters that influence the model
    concat_x_and_t = False # False works better with a simple architecture
    feed_forward_kernel = True
    # The Unet with one channel kernel doesn't work well with the gaussian data
    hidden_units = [256, 512, 256]
    noise_time_steps = 128  
    time_dim_emb = 64  # >=32 works well
    experiment_number = '0x'
    model_name = f"diffusion_model_gaussian_{experiment_number}"
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat_x_and_t}"

    loss_name = 'loss_gaussian_' + experiment_number
    original_data_name = 'original_data_gaussian_' + experiment_number
    sample_image_name = 'gen_samples_gaussian_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_gaussian_' + experiment_number
    inpainted_data_name = 'inpainted_data_gaussian_' + experiment_number

    #!WEIGHTS AND BIASES---------------------------------------------------------------- 
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionInpaintingGaussian", name=experiment_number)

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
    dataloader = dataset_generator.get_dataloader(batch_size=64, with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape()
    
    # Plot the original data
    dataset_generator.plot_data(original_data_name, save_locally=save_plots_locally, save_wandb=wandb_track)
    
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
                            hidden_units=hidden_units)
        # DDPM training
        train_losses = diffusion.train(dataloader=dataloader,
                                       learning_rate=learning_rate, 
                                       epochs=epochs,
                                       beta_ema=beta_ema)
        plot_loss(train_losses, loss_name, save_wandb=wandb_track, save_locally=save_plots_locally)
        diffusion.save_model_pickle(filename=model_name, 
                                    ema_model=True)
        # DDPM sampling
        sampled_data = diffusion.sample(samples=samples)[0].cpu().numpy()
        
        # Plot the sampled distribution
        plot_generated_samples(sampled_data, filename=sample_image_name, save_locally=save_plots_locally, save_wandb=wandb_track)

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
    plot_data_to_inpaint(x, mask, data_to_inpaint_name, save_locally=save_plots_locally, save_wandb=wandb_track)
    plot_generated_samples(inpainted_data, inpainted_data_name, save_locally=save_plots_locally, save_wandb=wandb_track)

    if wandb_track:
        wandb.finish()

def main_sum_categorical_data():
    
    # Are we going to use Weights and Biases?
    wandb_track = True
    save_plots_locally = False
    
    #!ARGUMENTS AND HYPERPARAMETERS-----------------------------------------------------
    # Hyperparameters that don't influence the model too much
    beta_ema = 0.95
    samples = 1000
    learning_rate = 1e-3
    epochs = 128
    inpaint_resampling = 20
    dropout_rate = 0.05
    
    # Dataset parameters
    threshold = 15
    structure = [2, 3, 5, 7, 11]
    
    #  Hyperparameters that influence the model
    time_dim_emb = 64  
    feed_forward_kernel = True 
    unet = False
    
    concat_x_and_t = True # True + 2*hidden_units works "well"
    input_dim_hidden = sum(structure) + time_dim_emb*concat_x_and_t
    hidden_units = [input_dim_hidden*2] # this works "well"
    # the diffusion model operates with logits, whose shape is sum(structure)
    
    # concat_x_and_t = False # also works "well"
    # input_dim_hidden = sum(structure) + time_dim_emb*concat_x_and_t
    # hidden_units = [input_dim_hidden*4, input_dim_hidden*9, input_dim_hidden*4] 
    #             # ~ [128, 256, 128]
    
    noise_time_steps = 128  
    experiment_number = '0x'
    only_anomalies = False
    model_name = f"diffusion_model_categorical_{experiment_number}"
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat_x_and_t}"
    # architecture_comment = f"2 encoders and decoders | concat(x, t): {concat_x_and_t}"
    
    loss_name = 'loss_categorical_' + experiment_number
    original_data_name = 'original_data_categorical_' + experiment_number
    sample_image_name = 'gen_samples_categorical_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_categorical_' + experiment_number
    inpainted_data_name = 'inpainted_data_categorical_' + experiment_number
    classifier_name = 'classifier_ddpm_sum_categorical_' + experiment_number

    #!WEIGHTS AND BIASES----------------------------------------------------------------
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionInpaintingCategorical", name=experiment_number)

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
                            'dropout_rate': dropout_rate,
                            'only_anomalies)': only_anomalies,
        })
    
    #!DATASET TO TRAIN THE MODEL---------------------------------------------------------
    size = 3000
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    dataset_train_ddpm = dataset_generator.generate_dataset(remove_anomalies=True)
    dataloader = dataset_generator.get_dataloader(batch_size=64, with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape() 

    # Plot the original distribution
    # Later we plot the generated samples to see compare if the distribution is similar
    plot_categories(dataset_train_ddpm['indices'], structure, original_data_name, 
                    save_locally=save_plots_locally, save_wandb=wandb_track)
    
    #!CLASSIFIER TO ASSESS HOW GOOD THE DIFFUSION MODEL CAPTURES THE DISTRIBUTION-------
    dataloader_classifier = dataset_generator.get_classifier_dataloader(training_prop=0.7, batch_size=64)
    classifier = ClassificationModel()
    classifier.load_model_pickle(classifier_name)
    
    if classifier.model is None:
        classifier.reset(dataloader_classifier.dataset.dataset.tensors[0].shape[1], 10)
        classifier.train(dataloader_classifier, n_epochs=200, learning_rate=1e-3, 
                         model_name=classifier_name)

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
                            unet=unet, 
                            dropout_rate=dropout_rate)  
        # DDPM training
        train_losses = diffusion.train(dataloader=dataloader,
                                       learning_rate=learning_rate, 
                                       epochs=epochs,
                                       beta_ema=beta_ema, 
                                       wandb_track=wandb_track)
        plot_loss(train_losses, loss_name, save_wandb=wandb_track, save_locally=save_plots_locally)
        diffusion.save_model_pickle(filename=model_name, 
                                    ema_model=True)
        # DDPM sampling
        sampled_logits = diffusion.sample(samples=samples)[0]
        sampled_data = dataset_generator.logits_to_values(sampled_logits).astype(int)
        
        dissimilarity = compute_divergence(dataset_train_ddpm['indices'].numpy().astype(int),
                                           sampled_data)
        
        print(f"Dissimilarity between the original and the sampled distribution: {dissimilarity:.2%}\n")
        
        classifier.model.eval()
        with torch.no_grad():
            prediction = classifier.model(torch.tensor(sampled_data, dtype=torch.float64))
            y = np.round(prediction.numpy().flatten())
            percentage_anomalies = y.mean()
        
        print(f"Percentage of anomalies in the generated samples: {percentage_anomalies:.2%}\n")
        
        # Plot the sampled distribution
        plot_categories(sampled_data, structure, sample_image_name, save_wandb=wandb_track, 
                        save_locally=save_plots_locally)

    #!DATA TO INPAINT-------------------------------------------------------------------
    # generate inpainting samples
    size = 1500
    dataset_generator = SumCategoricalDataset(size, structure, threshold)
    _ = dataset_generator.generate_dataset(remove_anomalies=False, only_anomalies=only_anomalies)
    data_to_inpaint = dataset_generator.get_features_with_mask(mask_anomaly_points=False,
                                                               mask_one_feature=True, 
                                                               label_values_mask=True, 
                                                               )
    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']
    
    #!INPAINTING------------------------------------------------------------------
    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(original=x,
                                        mask=mask, 
                                        resampling_steps=inpaint_resampling)
    inpainted_data = dataset_generator.logits_to_values(inpainted_data)

    # Distribution of the data to inpaint and the inpainted data
    plot_categories(data_to_inpaint['indices'], structure, data_to_inpaint_name, 
                    save_locally=save_plots_locally, save_wandb=wandb_track)
    plot_categories(inpainted_data, structure, inpainted_data_name, 
                        save_locally=save_plots_locally, save_wandb=wandb_track)
        
    #!METRICS--------------------------------------------------------------------------
    # Percentage of anomalies in the inpainted data
    classifier.model.eval()
    percentage_anomalies_inpainted_classifier = None
    with torch.no_grad():
        prediction = classifier.model(torch.tensor(inpainted_data, dtype=torch.float64))
        y = np.round(prediction.numpy().flatten())
        percentage_anomalies_inpainted_classifier = y.mean()
    
    # Count the number of original anomalies
    y = data_to_inpaint['y'].numpy().squeeze().astype(bool)
    number_anomalies = np.sum(y)
    percentage_anomalies_before = np.mean(y)
    
    inpaint_detailed = element_wise_label_values_comparison(data_to_inpaint['indices'], 
                                                    inpainted_data, 
                                                    data_to_inpaint['values_mask'])
    num_rows_differ, known_values, total_wrongly_changed_values = inpaint_detailed

    # Dissimilarity between the original and the inpainted distribution
    dissimilarity = compute_divergence(data_to_inpaint['indices'].numpy().astype(int),
                                       inpainted_data)

    # Count the number of times the threshold is exceeded
    y_after = inpainted_data.sum(axis=1) > threshold
    number_remaining_anomalies = np.sum(y_after)
    percentage_anomalies_after = np.mean(y_after)
    percentage_change = (number_remaining_anomalies - number_anomalies) / number_anomalies
    right_changes = (y & ~y_after).sum().item()
    wrong_changes = (~y & y_after).sum().item()
    
    # plot_agreement_disagreement_transformation(y, y_after, inpainted_data_name)
    
    #!SUMMARY--------------------------------------------------------------------------
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=50)
    table.add_column("Value")
    
    table.add_row("Anomalies before inpainting / Total", f"{number_anomalies} / {size}")
    table.add_row("Remaining anomalies / Total", f"{number_remaining_anomalies} / {size}")
    table.add_row("Correct changes", f"{right_changes}")
    table.add_row("Wrong changes", f"{wrong_changes}")
    table.add_row("Percentage of anomalies before inpainting", f"{percentage_anomalies_before:.2%}")
    table.add_row("Percentage of anomalies after inpainting", f"{percentage_anomalies_after:.2%}")
    table.add_row("Percentage change (-100% desired)", f"{percentage_change:.2%}%")
    table.add_row("Percentage of classified anomalies after inpainting", f"{percentage_anomalies_inpainted_classifier:.2%}")
    table.add_row("Number of rows wrongly modified", f"{num_rows_differ}({size})")
    table.add_row("Number of known values wrongly modified", f"{total_wrongly_changed_values}({known_values})")
    table.add_row("Dissimilarity between the original and the inpainted distribution", f"{dissimilarity:.2%}")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    if wandb_track:
        wandb.finish()  

if __name__ == '__main__':
    # main_gaussian_data()
    main_sum_categorical_data()
    
