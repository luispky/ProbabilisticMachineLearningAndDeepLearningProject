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
    args.epochs = 50
    args.lr = 1e-3
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.samples = 1000
    
    # define the components of the DDPM model: dataset, scheduler, model, EMA class
    dataset = Dataset()
    # dataloader = dataset.generate_data(with_labels=True)
    dataloader = dataset.generate_data(with_labels=False)
    dataset_shape = dataset.get_dataset_shape()
    noise_time_steps = 100
    scheduler = LinearNoiseScheduler(noise_timesteps=noise_time_steps, dataset_shape=dataset_shape)
    time_dim_embedding = 100
    model = NoisePredictor(dataset_shape = dataset_shape, time_dim=time_dim_embedding, num_classes=2)
    ema = EMA(beta=0.995)
    
    # Instantiate the DDPM model
    diffusion = DDPM(scheduler, model, args)
    
    # train the model
    diffusion.train(dataloader, ema)
    
    # save model
    # diffusion.save_model(diffusion.model, 'ddpm_model_03')
    
    # generate samples and 
    # bring labels and samples to the cpu
    # labels = torch.randint(0, 2, (args.samples,)).to(args.device)
    # labels = labels.cpu().numpy()
    labels = None
    samples = diffusion.sample(diffusion.model, labels)
    samples = samples.cpu().numpy()
    
    # save the generated samples
    # save_plot_generated_samples('generated_samples_19', samples, labels=None)
    save_plot_generated_samples('generated_samples_23', samples, labels=None) 

def test():
    dataset = Dataset()
    dataset.plot_data()
    
if __name__ == '__main__':
    main()