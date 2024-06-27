import torch
import argparse
import sys
import os
import wandb

# * Run from the scripts directory with:
# Add the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath('..'))
# python run_diffusion.py

# * Run from the parent directory with:
# python -m scripts.run_diffusion 
# not recommended because it will create another directory for the plots

from src import DDPM, NoisePredictor, Dataset, LinearNoiseScheduler, EMA, save_plot_generated_samples, plot_loss

def main():
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
    
    experiment_number = '27'
    architecture_comment = '4 layers, ins: 2, 32, 64, 32 | sum x and t'
    #  Hyperparameters that influence the model
    noise_time_steps = 128 # 128 good value, try 256
    time_dim_embedding = 32 # >=32 work well
    
    save_model = False
    
    sample_image_name = 'gen_samples_'+ experiment_number
    model_name = 'ddpm_model_' + experiment_number
    loss_name = 'loss_' + experiment_number
    
    # Initialize a new wandb run
    wandb.init(project="Diffusion_Model", name=experiment_number)

    
    # Add all the hyperparameters to wandb
    wandb.config.update({"architecture": architecture_comment,
                         "sampler": sampler_comment, 
                        'noise_time_steps': noise_time_steps,
                        'time_dim_embedding': time_dim_embedding,
                        'epochs': args.epochs,
                        'learning_rate': args.lr,
                        'cfg_strength': args.cfg_strength,
                        'beta_ema': beta_ema,
                        'samples': args.samples,
                        'num_classes': num_classes,
    })
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    ema = EMA(beta=beta_ema)
    dataset = Dataset()
    
    if with_labels:
        dataloader = dataset.generate_data(with_labels=True)
    else:
        dataloader = dataset.generate_data(with_labels=False)
    
    dataset_shape = dataset.get_dataset_shape()
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    model = NoisePredictor(dataset_shape = dataset_shape, time_dim=time_dim_embedding, num_classes=num_classes)
    
    # Instantiate the DDPM model
    diffusion = DDPM(scheduler, model, args)
    
    # train the model
    train_losses = diffusion.train(dataloader, ema)
    
    # Plot the loss
    plot_loss(train_losses, loss_name)
    
    # save model
    if save_model:
        diffusion.save_model(diffusion.model, model_name)
    
    # generate samples and 
    # bring labels and samples to the cpu
    if with_labels:
        labels = torch.randint(0, num_classes, (args.samples,)).to(args.device)
    else:
        labels = None
        
    samples = diffusion.sample(diffusion.ema_model, labels, args.cfg_strength)
    samples = samples.cpu().numpy()
    
    if with_labels:
        labels = labels.cpu().numpy()
    
    # save the generated samples
    if with_labels:
        save_plot_generated_samples(sample_image_name, samples, labels=labels)
    else:   
        save_plot_generated_samples(sample_image_name, samples, labels=None) 

    wandb.finish()

def test():
    dataset = Dataset()
    dataset.plot_data()
    
if __name__ == '__main__':
    main()