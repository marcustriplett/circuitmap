import argparse
import numpy as np
import circuitmap as cm
from circuitmap.simulation import simulate
from circuitmap import NeuralDemixer
from datetime import date
import _pickle as cpickle # pickle compression
import bz2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--N')
	parser.add_argument('--nreps')
	parser.add_argument('--connection_prob')
	parser.add_argument('--spont_prob')
	parser.add_argument('--ntars')
	parser.add_argument('--trials')
	parser.add_argument('--demixer')
	parser.add_argument('--token')
	parser.add_argument('--out')
	parser.add_argument('--weights')
	parser.add_argument('--weight_index')
	args = parser.parse_args()

	N = int(args.N)
	nreps = int(args.nreps)
	ntars = int(args.ntars)
	spont_prob = float(args.spont_prob)
	connection_prob = float(args.connection_prob)
	trials = int(args.trials)
	design = args.design

	if (args.weights is not None) and (args.weight_index is not None):
		weights_list = np.load(args.weights)
		weight_indx = int(args.weight_indx)
		weights = weights_list[weight_indx] # load weights
	else:
		weights = None

	powers = np.array([45, 55, 65])
	msrmp = 0.4

	# Simulate data
	sim = simulate(H=ntars, N=N, nreps=nreps, spont_prob=spont_prob, connection_prob=connection_prob,
		design='blockwise', trials=trials, powers=powers, max_power_min_spike_rate=msrmp, batch_size=100)

	# Denoise traces
	demix = NeuralDemixer(path=args.demixer, device='cpu')
	psc_dem = demix(sim['psc'])
	stim_matrix = sim['stim_matrix']
	y = np.trapz(psc_dem, axis=1)
	K = psc_dem.shape[0]

	# Configure fit options
	iters = 50
	seed = 1
	minimum_spike_count = 3
	lam_mask_fraction = 0.025

	fit_options_caviar = { 
		'iters': iters,
		'seed': seed,
		'msrmp': msrmp,
		'minimum_spike_count': minimum_spike_count,
		'lam_mask_fraction': lam_mask_fraction
	}

	fit_options_sns = { 
		'iters': iters,
		'seed': seed,
		'minimum_spike_count': minimum_spike_count,
		'lam_mask_fraction': lam_mask_fraction
	}

	model_caviar = cm.Model(N, priors=priors_caviar)
	model_caviar.fit(psc_dem, stim_matrix, fit_options=fit_options_caviar, method='caviar')

	model_sns = cm.Model(N, priors=priors_caviar)
	model_sns.fit(psc_dem, stim_matrix, fit_options=fit_options_sns, method='cavi_sns')

	d = {
		'N': N,
		'nreps': nreps,
		'ntars': ntars,
		'spont_prob': spont_prob,
		'connection_prob': connection_prob,
		'weights': sim['weights']
	}

	d['caviar'] = model_caviar.state['mu']
	d['sns_mu'] = model_sns.state['mu']
	d['sns_alpha'] = model_sns.state['alpha']

	tar_matrix = np.zeros_like(stim_matrix)
	tar_matrix[stim_matrix > 0] = 1
	w_cosamp = cm.optimise.cosamp(tar_matrix.T, np.trapz(psc_dem, axis=-1), np.sum(sim['weights'] != 0))[0]

	d['cosamp'] = w_cosamp

	out = args.out
	if out[-1] != '/': out += '/'

	with bz2.BZ2File('%ssim_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontprob%.3f'%(out, N, K, ntars, nreps, connection_prob, spont_prob) \
		+ '_trial%s_%s.pkl'%(args.token, date.today().__str__()), 'wb') as savefile:
		cpickle.dump(d, savefile)
