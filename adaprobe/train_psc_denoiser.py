import yaml
import sys
import os

# from adaprobe.psc_denoiser import NeuralDenoiser
from psc_denoiser import NeuralDenoiser

if __name__ == '__main__':
	with open(sys.argv[1]) as file:
		config = yaml.full_load(file)

	# Define denoiser object
	denoiser = NeuralDenoiser(
		n_layers=config['n_layers'], 
		channels=config['channels'], 
		kernel_size=config['kernel_size'], 
		padding=config['padding']
	)

	# Generate training data
	denoiser.generate_training_data(trial_dur=900, size=config['size'], gp_scale=0.03, min_delta=160, 
		delta_upper=140, next_min_delta=300, next_delta_upper=600)

	if not os.path.isdir(config['save_path']):
		os.mkdir(config['save_path'])

	# Train
	denoiser.train(
		epochs=config['epochs'], 
		save_path=config['save_path'], 
		save_every=config['save_every']
	)