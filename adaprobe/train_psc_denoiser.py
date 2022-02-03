import argparse
import numpy as np
from adaprobe.psc_denoiser import NeuralDenoiser

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size')
	parser.add_argument('--epochs')
	parser.add_argument('--templates')
	parser.add_argument('--pretrained')
	parser.add_argument('')
	args = parser.parse_args()

	size = int(args.size)
	epochs = int(args.epochs)

	# Optionally load negative training templates
	if args.templates is None:
		templates = None
	else:
		templates = np.load(args.templates)

	# Optionally load pretrained model
	if args.pretrained is None:
		denoiser = NeuralDenoiser()
	else:
		denoiser = NeuralDenoiser(path=args.pretrained)

	# Params for chrome2f + interneuron -> pyramidal currents
	# tau_diff_lower=150
	# tau_diff_upper=340

	# Params for chrome1 + pyramidal -> pyramidal currents
	tau_diff_lower=60
	tau_diff_upper=80

	denoiser.generate_training_data(trial_dur=900, size=size, gp_scale=0.045, delta_lower=160,
								delta_upper=400, next_delta_lower=400, next_delta_upper=899,
								prev_delta_upper=150, tau_diff_lower=150, tau_diff_upper=340,
								tau_r_lower=10, tau_r_upper=40, noise_std_lower=0.001,
								noise_std_upper=0.02, gp_lengthscale=45, templates=templates)
	denoiser.train(epochs=epochs)