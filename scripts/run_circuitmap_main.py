import argparse
import numpy as np
import circuitmap as cm
from circuitmap.neural_waveform_demixing import NeuralDemixer
from scipy.io import savemat, loadmat
import yaml
from pathlib import Path

# configure JAX memory preallocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

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
	psc = f['psc']
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
	model.fit(psc_dem, stim_matrix, method='caviar', fit_options={'msrmp': msrmp, 'save_histories': False})

	# save results
	out = args.out
	if out[-1] != '/':
		out += '/'

	base = Path(args.data).stem
	save_name = out + base + '_cmap.' + save_fmt

	if save_fmt == 'mat':
		savemat(save_name, {'weights': model.state['mu'], 'weight_uncertainty': model.state['beta'],
			'spikes': model.state['lam']})
	else:
		np.savez(save_name + save_fmt, weights=model.state['mu'], weight_uncertainty=model.state['beta'], 
			spikes=model.state['lam'])
