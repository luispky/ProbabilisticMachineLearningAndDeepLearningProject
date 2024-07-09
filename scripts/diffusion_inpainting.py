import sys
import os
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch
import wandb
from src.utils import plot_loss
from src.utils import SumCategoricalDataset
from src.denoising_diffusion_pm import DDPM
from src.utils import plot_categories, ClassificationModel, compute_arrays_agreements
from datasets.generate_data import compute_divergence
from src.utils import RealDataset
from rich.console import Console
from rich.table import Table

# set default type to avoid problems with gradient
DEFAULT_TYPE = torch.float64
torch.set_default_dtype(DEFAULT_TYPE)
torch.set_default_device('cpu')


def main():
    
    # Are we going to use Weights and Biases?
    wandb_track = False
    save_plots_locally = True
    
    #!DATASET TO TRAIN THE MODEL---------------------------------------------------------
    dataset_generator = RealDataset('bank-additional-ful-nominal')
    dataset_train_ddpm = dataset_generator.generate_dataset(remove_anomalies=True)
    dataloader = dataset_generator.get_dataloader(batch_size=64, with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape()
    
    
    #!ARGUMENTS AND HYPERPARAMETERS-----------------------------------------------------
    # Hyperparameters that don't influence the model too much
    beta_ema = 0.95
    samples = 1000
    learning_rate = 1e-3
    epochs = 128
    inpaint_resampling = 20
    dropout_rate = 0.05
    
    # Dataset parameters
    degrees_of_freedom_categories = dataset_generator.get_degrees_of_freedom_categories() 
    #  Hyperparameters that influence the model
    time_dim_emb = 64  
    feed_forward_kernel = True 
    unet = False
    
    concat_x_and_t = True # True + 2*hidden_units works "well"
    input_dim_hidden = degrees_of_freedom_categories + time_dim_emb*concat_x_and_t
    hidden_units = [input_dim_hidden*2] 
    
    # concat_x_and_t = False # also works "well"
    # input_dim_hidden = degrees_of_freedom_categories + time_dim_emb*concat_x_and_t
    # hidden_units = [input_dim_hidden*4, input_dim_hidden*9, input_dim_hidden*4] 
    # hidden_units = [128, 256, 128]
    
    noise_time_steps = 128  
    experiment_number = '0x'
    only_anomalies = True
    model_name = f"diffusion_model_categorical_{experiment_number}"
    architecture_comment = f"hidden_units: {hidden_units} | concat(x, t): {concat_x_and_t}"
    
    loss_name = 'loss_categorical_' + experiment_number
    classifier_name = 'classifier_ddpm_sum_categorical_' + experiment_number

    #!WEIGHTS AND BIASES----------------------------------------------------------------
    if wandb_track:
        # Initialize a new wandb run
        wandb.init(project="DiffusionInpaintingBank", name=experiment_number)

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
    
    #!CLASSIFIER TO ASSESS HOW GOOD THE DIFFUSION MODEL CAPTURES THE DISTRIBUTION-------
    dataloader_classifier = dataset_generator.get_classifier_dataloader(training_prop=0.7, batch_size=64)
    classifier = ClassificationModel()
    classifier.load_model_pickle(classifier_name)
    
    if classifier.model is None:
        input_size = dataloader_classifier.dataset.dataset.tensors[0].shape[1]
        classifier.reset(input_size, 5*input_size)
        classifier.train(dataloader_classifier, n_epochs=200, learning_rate=1e-3, 
                         model_name=classifier_name, weight_decay=1e-5)

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
    
    #!DATA TO INPAINT-------------------------------------------------------------------
    _ = dataset_generator.generate_dataset(remove_anomalies=False, only_anomalies=only_anomalies)
    data_to_inpaint = dataset_generator.get_features_with_mask()                                                         
    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']
    
    #!INPAINTING------------------------------------------------------------------
    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(original=x,
                                        mask=mask, 
                                        resampling_steps=inpaint_resampling)
    inpainted_data = dataset_generator.logits_to_values(inpainted_data)

    #!METRICS--------------------------------------------------------------------------
    # Percentage of anomalies in the inpainted data
    classifier.model.eval()
    percentage_anomalies_inpainted_classifier = None
    y_after_classifier = None
    with torch.no_grad():
        prediction = classifier.model(torch.tensor(inpainted_data, dtype=torch.float64))
        y_after_classifier = np.round(prediction.numpy().flatten()).astype(bool)
        percentage_anomalies_inpainted_classifier = y_after_classifier.mean()
    
    # Count the number of original anomalies
    y = data_to_inpaint['mask'].numpy().squeeze().astype(bool)
    size = y.size   
    number_anomalies = np.sum(y)
    percentage_anomalies_before = np.mean(y)
    
    agreements = compute_arrays_agreements(data_to_inpaint['indices'], inpainted_data)
    mean = agreements['mean']
    meadian = agreements['median']
    std = agreements['std']

    # Dissimilarity between the original and the inpainted distribution
    dissimilarity = compute_divergence(data_to_inpaint['indices'].numpy().astype(int),
                                       inpainted_data.astype(int))

    # Count the number of times the threshold is exceeded
    number_remaining_anomalies = np.sum(y_after_classifier)
    percentage_change = (number_remaining_anomalies - number_anomalies) / number_anomalies
    right_changes = (y & ~y_after_classifier).sum().item()
    wrong_changes = (~y & y_after_classifier).sum().item()
    
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
    table.add_row("Percentage change (-100% desired)", f"{percentage_change:.2%}%")
    table.add_row("Percentage of classified anomalies after inpainting", f"{percentage_anomalies_inpainted_classifier:.2%}")
    table.add_row(f"Mean values agree per instance (columns)", f"{mean:.2f} ({data_to_inpaint['indices'].shape[1]})")
    table.add_row("Median values agree per instance", f"{meadian:.2f}")
    table.add_row("Standard deviation values agree per instance", f"{std:.2f}")
        
    table.add_row("Dissimilarity between the original and the inpainted distribution", f"{dissimilarity:.2%}")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    if wandb_track:
        wandb.finish()
    
if __name__ == '__main__':
    main()
    
