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
    num_classes = 2
    beta_ema = 0.995
    args.samples = 1000
    args.cfg_strength = 2.0
    time_dim_embedding = 100
    
    noise_time_steps = 100
    args.epochs = 50
    args.lr = 1e-3
    
    with_labels = False
    image_name = 'generated_samples_'+'24'
    
    save_model = False
    model_name = 'ddpm_model_' + '03'
    
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
    diffusion.train(dataloader, ema)
    
    # save model
    if save_model:
        diffusion.save_model(diffusion.model, model_name)
    
    # generate samples and 
    # bring labels and samples to the cpu
    if with_labels:
        labels = torch.randint(0, num_classes, (args.samples,)).to(args.device)
        labels = labels.cpu().numpy()
    else:
        labels = None
        
    samples = diffusion.sample(diffusion.model, labels, args.cfg_strength)
    samples = samples.cpu().numpy()
    
    # save the generated samples
    if with_labels:
        save_plot_generated_samples(image_name, samples, labels=labels)
    else:   
        save_plot_generated_samples(image_name, samples, labels=None) 

def test():
    dataset = Dataset()
    dataset.plot_data()
    
if __name__ == '__main__':
    main()