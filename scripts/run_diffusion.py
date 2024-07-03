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
    # define the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Not required for the inpainting task
    with_labels = False
    args.cfg_strength = 3.0
    args.num_classes = 2

    # Hyperparameters that don't influence the model too much
    beta_ema = 0.999
    args.samples = 1000
    args.lr = 1e-3
    sampler_comment = 'ema_model'
    args.epochs = 64
    
    experiment_number = '30'
    architecture_comment = '4 layers, ins: 2, 32, 64, 32 | sum x and t'
    #  Hyperparameters that influence the model
    noise_time_steps = 128  # 128 good value, try 256
    time_dim_embedding = 64  # >=32 works well

    save_model = False

    sample_image_name = 'gen_samples_' + experiment_number
    model_name = 'ddpm_model_' + experiment_number
    loss_name = 'loss_' + experiment_number
    inpainted_data_name = 'inpainted_data_' + experiment_number

    # Initialize a new wandb run
    wandb.init(project="Diffusion_Model", name=experiment_number)

    # Add all the hyperparameters to wandb
    wandb.config.update({"architecture": architecture_comment,
                         "sampler": sampler_comment, 
                        'noise_time_steps': noise_time_steps,
                        'time_dim_embedding': time_dim_embedding,
                        'epochs': args.epochs,
                        'scheduler': 'linear',
                        'learning_rate': args.lr,
                        'cfg_strength': args.cfg_strength,
                        'beta_ema': beta_ema,
                        'samples': args.samples,
                        'num_classes': args.num_classes,
    })
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    ema = EMA(beta=beta_ema)
    means = [[-4, -4], [8, 8], [-4, 7]]
    covariances = [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]
    num_samples_per_distribution = [1000, 2000, 1500]
    dataset_generator = GaussianDataset()
    dataset_generator.generate_dataset(means, covariances, num_samples_per_distribution)
    dataloader = dataset_generator.get_dataloader()
    dataset_shape = dataset_generator.get_dataset_shape()
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    # scheduler = CosineNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(dataset_shape=dataset_shape, time_dim=time_dim_embedding, num_classes=args.num_classes)

    # Instantiate the DDPM modelcd 
    diffusion = DDPM(scheduler, model, args)

    # train the model
    train_losses = diffusion.train(dataloader, ema)

    # Plot the loss
    plot_loss(train_losses, loss_name)

    # save model
    if save_model:
        save_model_to_dir(diffusion.model, model_name)

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
    
    experiment_number = '05'
    architecture_comment = '5 layers, ins: 92, 64, 32, 92 | concat x and t'
    #  Hyperparameters that influence the model
    noise_time_steps = 128  # 128 good value, try 256
    time_dim_embedding = 64  # >=32 work well

    original_data_name = 'original_data_' + experiment_number
    sample_image_name = 'gen_samples_' + experiment_number
    model_name = 'ddpm_model_' + experiment_number
    loss_name = 'loss_' + experiment_number
    inpainted_data_name = 'inpainted_data_' + experiment_number
    data_to_inpaint_name = 'data_to_inpaint_' + experiment_number

    # Initialize a new wandb run
    wandb.init(project="DiffusionModelSumCategorical", name=experiment_number)

    # Add all the hyperparameters to wandb
    wandb.config.update({"architecture": architecture_comment,
                         "sampler": sampler_comment, 
                        'noise_time_steps': noise_time_steps,
                        'time_dim_embedding': time_dim_embedding,
                        'epochs': args.epochs,
                        'scheduler': 'linear',
                        'learning_rate': args.lr,
                        'beta_ema': beta_ema,
                        'samples': args.samples,
    })
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    ema = EMA(beta=beta_ema)
    size = 3000
    threshold = 15
    n_values = [2, 3, 5, 7, 11]
    dataset_generator = SumCategoricalDataset(size, n_values, threshold)
    _ = dataset_generator.generate_dataset(remove_anomalies=True, logits=True)
    dataloader = dataset_generator.get_dataloader(with_labels=False)
    dataset_shape = dataset_generator.get_dataset_shape()
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(dataset_shape=dataset_shape, time_dim=time_dim_embedding)
    prob_instance = Probabilities(n_values)
    
    print(f'Size of the dataset requested: {size} samples') 
    print(f'Training dataset size without anomalies: {len(dataloader.dataset)} samples')
    print(f'Logits to train min value: {dataloader.dataset.tensors[0].min()}')
    print(f'Logits to train max value: {dataloader.dataset.tensors[0].max()}\n')
    
    # plot the train label encoded values
    plot_categories(dataset_generator.label_values, n_values, original_data_name)
    
    # Instantiate the DDPM model
    diffusion = DDPM(scheduler, model, args)

    # train the model
    train_losses = diffusion.train(dataloader, ema)
    
    # Plot the loss
    plot_loss(train_losses, loss_name)
    
    # generate samples
    samples_logits = diffusion.sample(diffusion.ema_model)[0]
    print(f'Samples logits min value: {samples_logits.min()}')
    print(f'Samples logits max value: {samples_logits.max()}\n')
    samples = prob_instance.logits_to_values(samples_logits.cpu().numpy())
    
    # save the generated samples
    plot_categories(samples, n_values, sample_image_name)
    
    # generate inpainting samples
    size = 1500
    dataset_generator = SumCategoricalDataset(size, n_values, threshold)
    data_to_inpaint = dataset_generator.get_features_with_mask(label_values_mask=True)
    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']   
    plot_categories(dataset_generator.label_values, n_values, data_to_inpaint_name)
    
    print(f'Size if the dataset to inpaint: {x.shape[0]} samples')
    print(f'Logits to inpaint min value: {x.min()}')
    print(f'Logits to inpaint max value: {x.max()}\n')

    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(diffusion.ema_model, x, mask)
    print(f'Logits inpainted min value: {inpainted_data.min()}')
    print(f'Logis inpainted max value: {inpainted_data.max()}\n') 
    
    inpainted_data = prob_instance.logits_to_values(inpainted_data.cpu().numpy())
    plot_categories(inpainted_data, n_values, inpainted_data_name) 
    
    # Count the number of original anomalies
    y = data_to_inpaint['y'].numpy().squeeze()
    number_anomalies = np.sum(y)
    
    # Count the number of times the threshold is exceeded
    y_after = inpainted_data.sum(axis=1) > threshold
    number_remaining_anomalies = np.sum(y_after)
    percentage_change = (number_remaining_anomalies - number_anomalies) / number_anomalies * 100
    right_changes = (y & ~y_after).sum().item()
    wrong_changes = (~y & y_after).sum().item()
    
    inpaint_detailed = element_wise_label_values_comparison(data_to_inpaint['label_values'], 
                                                      inpainted_data, 
                                                      data_to_inpaint['values_mask'])
    num_rows_differ, known_values, total_wrongly_changed_values = inpaint_detailed
    
    # plot_agreement_disagreement_transformation(y, y_after, inpainted_data_name)
    
    # Summary of the inpainting process
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=40)
    table.add_column("Value")
    
    # print(f'Summary of the inpainting process...')
    # print(f'Data size: {size}')
    # print(f'Anomalies before inpainting / Total: {number_anomalies} / {size}')
    # print(f'Remaining anomalies / Total: {number_remaining_anomalies} / {size}')
    # print(f'Percentage change (-100% desired): {percentage_change:.2f}%')
    # print(f'Correct changes (balance): {right_changes} ({number_anomalies - right_changes})')
    # print(f'Wrong changes (balance): {wrong_changes} ({number_anomalies - wrong_changes})')
    # print(f'Number of rows wrongly modified: {num_rows_differ}({size})')
    # print(f'Number of known values wrongly modified: {total_wrongly_changed_values}({known_values})')
    
    table.add_row("Data size:", str(size))
    table.add_row("Anomalies before inpainting / Total:", f"{number_anomalies} / {size}")
    table.add_row("Remaining anomalies / Total:", f"{number_remaining_anomalies} / {size}")
    table.add_row("Percentage change (-100% desired):", f"{percentage_change:.2f}%")
    table.add_row("Correct changes (balance):", f"{right_changes} ({number_anomalies - right_changes})")
    table.add_row("Wrong changes (balance):", f"{wrong_changes} ({number_anomalies - wrong_changes})")
    table.add_row("Number of rows wrongly modified:", f"{num_rows_differ}({size})")
    table.add_row("Number of known values wrongly modified:", f"{total_wrongly_changed_values}({known_values})")

    console.print("[bold green]Summary of the inpainting process...[/bold green]")
    console.print(table)
    
    wandb.finish()

if __name__ == '__main__':
    # main_gaussian_data()
    main_sum_categorical_data()
    
