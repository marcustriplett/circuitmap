import argparse
import numpy as np
from circuitmap.neural_waveform_demixing import NeuralDemixer

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--size')
	parser.add_argument('--epochs')
	parser.add_argument('--templates')
	parser.add_argument('--pretrained')
	parser.add_argument('--learning_rate')
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
		demixer = NeuralDemixer()
	else:
		demixer = NeuralDemixer(path=args.pretrained)

	if args.learning_rate is None:
		learning_rate = 1e-2
	else:
		learning_rate = float(args.learning_rate)

	# Params for chrome2f + interneuron -> pyramidal currents
	# tau_r_lower = 10
	# tau_r_upper = 40
	# tau_diff_lower = 150
	# tau_diff_upper = 340

	# Params for chrome1 + pyramidal -> pyramidal currents
	# tau_r_lower = 10
	# tau_r_upper = 40
	# tau_diff_lower = 60
	# tau_diff_upper = 120

	# Params for pyramidal -> PV synaptic currents
	# tau_r_lower = 3
	# tau_r_upper = 20
	# tau_diff_lower = 20 - tau_r_lower
	# tau_diff_upper = 110 - tau_r_upper

	# Params for chrome2s pyramidal -> pyramidal currents (Emx-Cre)
	# tau_r_lower = 20
	# tau_r_upper = 40
	# tau_diff_lower = 55 - tau_r_lower
	# tau_diff_upper = 140 - tau_r_upper

	# Params for chrome2s pyramidal -> PV currents (Emx-Cre, extremely fast synaptic currents)
	# fast currents, rise = 8, decay = 9
	# slow currents, rise = 15, decay = 50
	tau_r_lower = 8
	tau_r_upper = 15
	tau_diff_lower = 9 - tau_r_lower
	tau_diff_upper = 50 - tau_r_upper


	demixer.generate_training_data(trial_dur=900, size=size, gp_scale=0.045, delta_lower=160,
								delta_upper=400, next_delta_lower=400, next_delta_upper=899,
								prev_delta_upper=150, tau_diff_lower=tau_diff_lower, 
								tau_diff_upper=tau_diff_upper, tau_r_lower=tau_r_lower, 
								tau_r_upper=tau_r_upper, noise_std_lower=0.001,
								noise_std_upper=0.02, gp_lengthscale=45, templates=templates)
	demixer.train(epochs=epochs, learning_rate=learning_rate)