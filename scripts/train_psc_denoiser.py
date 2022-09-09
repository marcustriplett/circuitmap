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
	parser.add_argument('--down_filter_sizes', nargs=4, type=int, default=(16, 32, 64, 128))
	parser.add_argument('--up_filter_sizes', nargs=4, type=int, default=(64, 32, 16, 4))

	# whether we add a gp to the target waveforms
	parser.add_argument('--add_target_gp', action='store_true')
	parser.add_argument('--no_add_target_gp', dest='add_target_gp', action='store_false')
	parser.set_defaults(add_target_gp=True)
	parser.add_argument('--target_gp_lengthscale', default=25)
	parser.add_argument('--target_gp_scale', default=0.01)

	# whether we use the linear onset in the training data
	parser.add_argument('--linear_onset_frac', type=float, default=0.5)

	# photocurrent shape args
	parser.add_argument('--O_inf_min', type=float, default=0.3)
	parser.add_argument('--O_inf_max', type=float, default=1.0)
	parser.add_argument('--R_inf_min', type=float, default=0.3)
	parser.add_argument('--R_inf_max', type=float, default=1.0)
	parser.add_argument('--tau_o_min', type=float, default=5)
	parser.add_argument('--tau_o_max', type=float, default=7)
	parser.add_argument('--tau_r_min', type=float, default=26)
	parser.add_argument('--tau_r_max', type=float, default=29)

	# photocurrent timing args
	parser.add_argument('--onset_jitter_ms', type=float, default=1.0)
	parser.add_argument('--onset_latency_ms', type=float, default=0.2)

	parser.add_argument('--templates_path', type=str)
	parser.add_argument('--templates_frac', type=float, default=0.2)
	
	args = parser.parse_args()
	unet_args = dict(
		down_filter_sizes=args.down_filter_sizes,
		up_filter_sizes=args.up_filter_sizes,
	)
	pc_shape_params = dict(
		O_inf_min=args.O_inf_min,
		O_inf_max=args.O_inf_max,
		R_inf_min=args.R_inf_min,
		R_inf_max=args.R_inf_max,
		tau_o_min=args.tau_o_min,
		tau_o_max=args.tau_o_max,
		tau_r_min=args.tau_r_min,
		tau_r_max=args.tau_r_max,
	)

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
		demixer = NeuralDemixer(unet_args=unet_args)
	else:
		demixer = NeuralDemixer(path=args.pretrained, unet_args=unet_args)

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
								noise_std_upper=0.02, gp_lengthscale=45, pc_templates=templates,
								convolve=convolve, sigma=sigma,
								target=args.target,
								pc_scale_min=args.pc_scale_min,
								pc_scale_max=args.pc_scale_max,
								prev_pc_fraction=args.prev_pc_fraction,
								pc_fraction=args.pc_fraction,
								next_pc_fraction=args.next_pc_fraction,
								save_path=args.dataset_save_path,
								pc_shape_params=pc_shape_params,
								onset_jitter_ms=args.onset_jitter_ms,
								onset_latency_ms=args.onset_latency_ms,
								add_target_gp=args.add_target_gp,
								target_gp_lengthscale=args.target_gp_lengthscale,
								target_gp_scale=args.target_gp_scale,
								linear_onset_frac=args.linear_onset_frac,
								templates_path=args.templates_path,
								templates_frac=args.templates_frac,
								)
	demixer.train(epochs=epochs, num_gpus=num_gpus)