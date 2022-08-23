import argparse
import numpy as np
from circuitmap.neural_waveform_demixing import NeuralDemixer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size')
	parser.add_argument('--epochs')
	parser.add_argument('--templates')
	parser.add_argument('--pretrained')
	parser.add_argument('--num_gpus', default=0)
	parser.add_argument('--target', type=str, default='demixed')
	parser.add_argument('--pc_scale_min', type=float, default=0.05)
	parser.add_argument('--pc_scale_max', type=float, default=10.0)
	parser.add_argument('--prev_pc_fraction', type=float, default=0.2)
	parser.add_argument('--pc_fraction', type=float, default=0.5)
	parser.add_argument('--next_pc_fraction', type=float, default=0.2)
	parser.add_argument('--dataset_save_path', type=str)
	args = parser.parse_args()

	size = int(args.size)
	epochs = int(args.epochs)
	num_gpus = int(args.num_gpus)

	# Optionally load negative training templates
	if args.templates is None:
		templates = None
	else:
		templates = np.load(args.templates)

	# Optionally load pretrained model
	if args.pretrained is None:
		demixer = NeuralDemixer()
	else:
		demixer = NeuralDemixer(path=args.pretrained)

	# Params
	tau_r_lower = 10
	tau_r_upper = 40

	# Params for chrome2f + interneuron -> pyramidal currents
	# tau_diff_lower = 150
	# tau_diff_upper = 340
	# convolve = False
	# sigma = 1

	# Params for chrome1 + pyramidal -> pyramidal currents
	tau_diff_lower = 60
	tau_diff_upper = 120
	convolve = False
	sigma = 30

	demixer.generate_training_data(trial_dur=900, size=size, gp_scale=0.045, delta_lower=160,
								delta_upper=400, next_delta_lower=400, next_delta_upper=899,
								prev_delta_upper=150, tau_diff_lower=tau_diff_lower, 
								tau_diff_upper=tau_diff_upper, tau_r_lower=tau_r_lower, 
								tau_r_upper=tau_r_upper, noise_std_lower=0.001,
								noise_std_upper=0.02, gp_lengthscale=45, templates=templates,
								convolve=convolve, sigma=sigma,
								target=args.target,
								pc_scale_min=args.pc_scale_min,
								pc_scale_max=args.pc_scale_max,
								prev_pc_fraction=args.prev_pc_fraction,
								pc_fraction=args.pc_fraction,
								next_pc_fraction=args.next_pc_fraction,
								save_path=args.dataset_save_path)
	demixer.train(epochs=epochs, num_gpus=num_gpus)