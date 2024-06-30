import sys
import os
sys.path.append(os.path.abspath('..'))
import torch
import numpy as np
import argparse
import wandb
from src.utils import GaussianDataset, LinearNoiseScheduler, EMA, save_plot_generated_samples, plot_loss
from src.utils import plot_data_to_inpaint, SumCategoricalDataset
from src.modules import NoisePredictor
from src.denoising_diffusion_pm import DDPM
from src.denoising_diffusion_pm import save_model_to_dir
from src.utils import Probabilities, plot_agreement_disagreement_transformation, plot_categories
import pandas as pd

def main_gaussian_data():
    # define the arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Not required for the inpainting task
    with_labels = False
    args.cfg_strength = 3.0
    num_classes = 2

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
    time_dim_embedding = 64  # >=32 work well

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
                        'num_classes': num_classes,
    })
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    ema = EMA(beta=beta_ema)
    means = [[-4, -4], [8, 8], [-4, 7]]
    covariances = [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]
    num_samples_per_distribution = [1000, 2000, 1500]
    dataset_generator = GaussianDataset()
    dataloader = dataset_generator.get_dataloader(means, covariances, num_samples_per_distribution)
    dataset_shape = dataset_generator.get_dataset_shape()
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    # scheduler = CosineNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(dataset_shape=dataset_shape, time_dim=time_dim_embedding, num_classes=num_classes)

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
    labels = torch.randint(0, num_classes, (args.samples,)) if with_labels else None

    samples = diffusion.sample(diffusion.ema_model, labels, args.cfg_strength)
    samples = samples.cpu().numpy()

    labels = labels.cpu().numpy() if with_labels else None

    # save the generated samples
    save_plot_generated_samples(samples, sample_image_name, labels=labels)

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
    save_plot_generated_samples(inpainted_data, inpainted_data_name)

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
    
    experiment_number = '01'
    architecture_comment = '4 layers, ins: 2, 32, 64, 32 | sum x and t'
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
    dataloader = dataset_generator.get_dataloader()
    dataset_shape = dataset_generator.get_dataset_shape()
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(dataset_shape=dataset_shape, time_dim=time_dim_embedding)
    prob_instance = Probabilities(n_values)
    
    # plot the train label encoded values
    df_train = pd.DataFrame(dataset_generator.label_values, columns=n_values)
    
    plot_categories(df_train, original_data_name)
    
    # Instantiate the DDPM model
    diffusion = DDPM(scheduler, model, args)

    # train the model
    train_losses = diffusion.train(dataloader, ema)

    # Plot the loss
    plot_loss(train_losses, loss_name)
    
    # generate samples
    samples = diffusion.sample(diffusion.ema_model)
    # clamp the inpainted data to the range [0, 1]
    # the normalize method of the Probabilities class only works with values in the range [0, 1]
    samples = torch.clamp(samples, 0, 1)
    samples = samples.cpu().numpy()
    
    # print the min and max values of the inpainted data
    print(f'Min value: {samples.min()}')
    print(f'Max value: {samples.max()}') 
    
    # normalize the inpainted data with the Probabilities class
    normalized_data = prob_instance.normalize(samples)
    one_hot_data = prob_instance.prob_to_onehot(normalized_data)
    samples_values = prob_instance.onehot_to_values(one_hot_data)
    
    df_samples = pd.DataFrame(samples_values, columns=n_values)

    # save the generated samples
    plot_categories(df_samples, sample_image_name)
    
    # generate inpainting samples
    size = 1500
    dataset_generator = SumCategoricalDataset(size, n_values, threshold)
    _ = dataset_generator.generate_dataset()
    data_to_inpaint = dataset_generator.get_features_with_mask()

    x, mask = data_to_inpaint['x'], data_to_inpaint['mask']
    
    df_values_to_inpaint = pd.DataFrame(dataset_generator.label_values, columns=n_values)
    plot_categories(df_values_to_inpaint, data_to_inpaint_name)
    
    # inpaint the masked data: probabilities
    inpainted_data = diffusion.inpaint(diffusion.ema_model, x, mask)
    # print the min and max values of the inpainted data
    print(f'Min value: {inpainted_data.min()}')
    print(f'Max value: {inpainted_data.max()}') 
    
    # clamp the inpainted data to the range [0, 1]
    # the normalize method of the Probabilities class only works with values in the range [0, 1]
    inpainted_data = torch.clamp(inpainted_data, 0, 1).cpu().numpy()
    
    # todo: I get assert np.all(s > 0), f'Zero sum: {s}' error
    # I have to check the sum of the inpainted data
    
    # normalize the inpainted data with the Probabilities class
    normalized_data = prob_instance.normalize(inpainted_data)
    one_hot_data = prob_instance.prob_to_onehot(normalized_data)
    inpainted_data = prob_instance.onehot_to_values(one_hot_data)
    
    # Count the number of original anomalies
    y = data_to_inpaint['y'].numpy().squeeze()
    number_anomalies = np.sum(y)
    
    # Count the number of times the threshold is exceeded
    y_after = inpainted_data.sum(axis=1) > threshold
    number_remaining_anomalies = np.sum(y_after)
    
    right_changes = (y & ~y_after).sum().item()
    wrong_changes = (~y & y_after).sum().item()
    
    # plot the inpainted data and the agreement/disagreement transformation
    df_inpaint = pd.DataFrame(dataset_generator.label_values, columns=n_values)
    plot_categories(df_inpaint, inpainted_data_name) 
    
    plot_agreement_disagreement_transformation(y, y_after, inpainted_data_name)
    
    print(f'Anomalies before inpainting: {number_anomalies}')
    print(f'Remaining anomalies: {number_remaining_anomalies}')
    print(f'Correct changes: {right_changes}')
    print(f'Wrong changes: {wrong_changes}')
    
    # todo: 
    # 1. compare labels that changed from 1 to 0 and vice versa
    # there should be two counters: 
    # one for the number of anomalies that changed from 1 to 0
    # another for the number of non-anomalies that changed to anomalies
    
    wandb.finish()

if __name__ == '__main__':
    # main_gaussian_data()
    main_sum_categorical_data()
    
