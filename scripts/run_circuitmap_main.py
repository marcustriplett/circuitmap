import argparse
import numpy as np
from circuitmap.neural_waveform_demixing import NeuralDemixer
import circuitmap as cm
from scipy.io import savemat, loadmat
import yaml

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--config')
	parser.add_argument('--out')

	args = parser.parse_args()

	# parse inputs
	ext = args.data[-4:]
	if ext == '.mat':
		load_func = loadmat
	elif ext == '.npy' or ext == '.npz':
		load_func = np.load
	else:
		raise Exception

	f = load_func(args.data)
	psc = f['pscs']
	stim_matrix = f['stimulus_matrix']

	# read demixer and msrmp from config file
	config = yaml.safe_load(open(args.config))
	save_fmt = config['save_fmt']

	assert save_fmt in ['mat', 'npy']

	# place demixer on cpu to avoid memory clash between pytorch and JAX
	demix = NeuralDemixer(path=config['demixer'], device='cpu')
	psc_dem = demix(psc)

	# setup circuitmap model
	msrmp = float(config['msrmp'])
	N = stim_matrix.shape[0]
	model = cm.Model(N)

	# fit model
	cm.fit(psc_dem, stim_matrix, method='caviar', fit_options={'msrmp': msrmp})

	# save results
	out = args.out
	if out[-1] != '/':
		out += '/'

	save_name = args.out + args.data[:-4] + '_cmap.' + save_fmt

	if save_fmt == 'mat':
		savemat(save_name, {'weights': cm.state['mu'], 'weight_uncertainty': cm.state['beta'],
			'spikes': cm.state['lam']})
	else:
		np.savez(save_name + save_fmt, weights=cm.state['mu'], weight_uncertainty=cm.state['beta'], 
			spikes=cm.state['lam'])
