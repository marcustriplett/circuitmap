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
	parser.add_argument('--design')
	parser.add_argument('--demixer')
	parser.add_argument('--token')
	parser.add_argument('--out')
	args = parser.parse_args()

	assert args.design in ['random', 'blockwise']

	N = int(args.N)
	nreps = int(args.nreps)
	ntars = int(args.ntars)
	spont_prob = float(args.spont_prob)
	connection_prob = float(args.connection_prob)
	trials = int(args.trials)
	design = args.design

	powers = np.array([45, 55, 65])

	minimax_spike_prob = 0.3

	# Simulate data
	sim = simulate(H=ntars, N=N, nreps=nreps, spont_prob=spont_prob, connection_prob=connection_prob,
		design=design, trials=trials, powers=powers, max_power_min_spike_rate=minimax_spike_prob, batch_size=100)

	# Denoise traces
	demix = NeuralDemixer(path=args.demixer, device='cpu')
	psc_dem = demix(sim['psc'])
	stim_matrix = sim['stim_matrix']
	y = np.trapz(psc_dem, axis=1)
	K = psc_dem.shape[0]

	# Configure fit options
	iters = 50
	sigma = 1
	seed = 1
	y_xcorr_thresh = 1e-2
	max_penalty_iters = 50
	warm_start_lasso = True
	verbose = False
	minimum_spike_count = 2
	num_mc_samples_noise_model = 100
	noise_scale = 0.5
	init_spike_prior = 0.5
	num_mc_samples = 500
	penalty = 2
	max_lasso_iters = 1000
	scale_factor = 0.75
	constrain_weights = 'positive'
	orthogonal_outliers = True
	lam_mask_fraction = 0.025

	# priors_mbcs = {
	# 	'beta': 3e0 * np.ones(N),
	# 	'mu': np.zeros(N),
	# 	'shape': np.ones(K),
	# 	'rate': 1e-1 * np.ones(K),
	# }

	# config priors
	phi_prior 		= np.c_[0.125 * np.ones(N), 5 * np.ones(N)]
	phi_cov_prior 	= np.array([np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])
	alpha_prior 	= 0.15 * np.ones(N)
	beta_prior 		= 3e0 * np.ones(N)
	mu_prior 		= np.zeros(N)
	sigma 			= 1e0

	priors_caviar = {
		'alpha': alpha_prior,
		'beta': beta_prior,
		'mu': mu_prior,
		'phi': phi_prior,
		'phi_cov': phi_cov_prior,
		'shape': 1.,
		'rate': sigma**2,
	}

	fit_options_caviar = { 
		'iters': iters,
		'num_mc_samples': num_mc_samples,
		'y_xcorr_thresh': y_xcorr_thresh,
		'seed': seed,
		'minimax_spk_prob': minimax_spike_prob,
		'scale_factor': scale_factor,
		'penalty': penalty,
		'minimum_spike_count': minimum_spike_count,
	}

	fit_options_sns = { 
		'iters': iters,
		'num_mc_samples': num_mc_samples,
		'seed': seed,
		'minimum_spike_count': minimum_spike_count,
	}

	# fit_options_mbcs = { 
	# 	'iters': iters,
	# 	'num_mc_samples': num_mc_samples,
	# 	'penalty': penalty,
	# 	'max_penalty_iters': max_penalty_iters,
	# 	'max_lasso_iters': max_lasso_iters,
	# 	'scale_factor': scale_factor,
	# 	'constrain_weights': constrain_weights,
	# 	'y_xcorr_thresh': y_xcorr_thresh,
	# 	'seed': seed,
	# 	'verbose': verbose,
	# 	'warm_start_lasso': warm_start_lasso,
	# 	'minimum_spike_count': minimum_spike_count,
	# 	'minimum_maximal_spike_prob': minimax_spike_prob,
	# 	'noise_scale': noise_scale,
	# 	'init_spike_prior': init_spike_prior,
	# 	'orthogonal_outliers': orthogonal_outliers,
	# 	'lam_mask_fraction': lam_mask_fraction,
	# 	'outlier_penalty': outlier_penalty,
	# 	'delay_spont_estimation': delay_spont_estimation,
	# }

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
		'design': design,
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

	with bz2.BZ2File('%ssim_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontprob%.3f_design%s'%(out, N, K, ntars, nreps, connection_prob, spont_prob, design) \
		+ '_trial%s_%s.pkl'%(args.token, date.today().__str__()), 'wb') as savefile:
		cpickle.dump(d, savefile)
