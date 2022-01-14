import argparse
from adaprobe.psc_denoiser import NeuralDenoiser

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size')
	parser.add_argument('--epochs')
	args = parser.parse_args()

	size = int(args.size)
	epochs = int(args.epochs)

	denoiser = NeuralDenoiser()
	denoiser.generate_training_data(trial_dur=900, size=size, gp_scale=0.045, delta_lower=160,
								delta_upper=400, next_delta_lower=400, next_delta_upper=899,
								prev_delta_upper=150, tau_diff_lower=2, tau_diff_upper=150,
								tau_r_lower=10, tau_r_upper=80)
	denoiser.train(epochs=epochs)