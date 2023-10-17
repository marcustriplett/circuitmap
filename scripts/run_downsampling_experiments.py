import numpy as np
import circuitmap as cm
from circuitmap import NeuralDemixer
import argparse
import h5py
import os
from scipy.io import loadmat

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--demixer')
	parser.add_argument('--msrmp')
	parser.add_argument('--out')
	parser.add_argument('--downsample_time') # in seconds
	parser.add_argument('--n_repeats')
	parser.add_argument('--design')
	parser.add_argument('--reader')
	parser.add_argument('--method')
	args = parser.parse_args()

	design = args.design
	assert design in ['single', 'multi']

	# Parse inputs and load data
	print('Loading file at ', args.data)
	if args.reader == 'h5py':
		data = h5py.File(args.data)
		stim_matrix_complete = np.array(data['stimulus_matrix']).T
		psc = np.array(data['pscs']).T

	elif args.reader == 'scipy':
		data = loadmat(args.data)
		stim_matrix_complete = data['stimulus_matrix']
		psc = data['pscs']

	elif args.reader == 'phorc':
		data = h5py.File(args.data)
		psc = np.array(data['pscs_subtracted']) # photocurrent-subtracted PSCs
		stim_matrix_complete = np.array(data['stimulus_matrix']).T

	else:
		raise Exception

	print('Data attributes:')
	print('\tPSC array dimension ', psc.shape)
	print('\tStimulus matrix dimension ', stim_matrix_complete.shape)

	demix = NeuralDemixer(path=args.demixer, device='cpu')

	stim_matrix_complete = stim_matrix_complete.astype(float) # cast to float
	N = stim_matrix_complete.shape[0]
	msrmp = float(args.msrmp)
	n_repeats = int(args.n_repeats)
	dstime = int(args.downsample_time) # in seconds
	stim_freq = 30 # Hz

	if design == 'single':
		locs = np.where(np.sum(stim_matrix_complete > 0, axis=0) == 1)[0]
	elif design == 'multi':
		locs = np.where(np.sum(stim_matrix_complete > 0, axis=0) > 1)[0]

	stim_matrix = stim_matrix_complete[:, locs]
	psc_dem = demix(psc)[locs]
	K = psc_dem.shape[0]

	# Begin downsampling
	ds_step = dstime * stim_freq
	dstrials = np.concatenate([np.arange(ds_step, K+1, ds_step), [K]])
	nsteps = len(dstrials)

	estimated_weights = np.zeros((n_repeats, nsteps, N))

	for st in range(nsteps):
		for r in range(n_repeats):
			trials = np.random.choice(K, dstrials[st], replace=False)
			this_stim = stim_matrix[:, trials]

			model = cm.Model(N)
			if args.method == 'caviar':
				this_psc = psc_dem[trials]
				model.fit(this_psc, this_stim, method='caviar', fit_options={'save_histories': False, 'tol': 0.005, 'msrmp': msrmp, 'fn_scan': True})

			elif args.method == 'cavi_sns':
				this_psc = psc[locs][trials] # no NWD for cavi_sns
				model.fit(this_psc, this_stim, method='cavi_sns', fit_options={'save_histories': False})

			else:
				raise Exception

			estimated_weights[r, st] = model.state['mu']

	out = args.out
	if out[-1] != '/': out += '/'
	base = os.path.basename(args.data)[:-4]
	np.savez(out + base + '_downsampling_weights_steptime%i_nreps%i_design%s_method%s'%(dstime, n_repeats, design, args.method), weights=estimated_weights)

