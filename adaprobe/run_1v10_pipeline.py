import argparse
import numpy as np
import adaprobe
from adaprobe.psc_denoiser import NeuralDenoiser
from datetime import date
import _pickle as cpickle # pickle compression
import bz2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--denoiser')
	parser.add_argument('--iters')
	parser.add_argument('--minimax_spike_prob')
	parser.add_argument('--token')
	parser.add_argument('--frac_data')
	args = parser.parse_args()

	if args.frac_data is None:
		frac_data = 1
	else:
		frac_data = float(args.frac_data)

	# Load data
	print(args.data)
	f = np.load(args.data)
	psc_single_tar, psc_multi_tar = [f[param] for param in ['psc_single_tar', 'psc_multi_tar']]
	stimulus_matrix_single_tar, stimulus_matrix_multi_tar = [f[param] for param in ['stimulus_matrix_single_tar', 'stimulus_matrix_multi_tar']]
	N, K_single = stimulus_matrix_single_tar.shape
	K_multi = stimulus_matrix_multi_tar.shape[-1]
	valid_trials = int(frac_data * K_multi)

	# Trim data based on frac_data arg
	psc_multi_tar = psc_multi_tar[:valid_trials]
	stimulus_matrix_multi_tar = stimulus_matrix_multi_tar[:, :valid_trials]
	K_multi = stimulus_matrix_multi_tar.shape[-1]

	# Denoise traces
	denoiser = NeuralDenoiser(path=args.denoiser)
	den_psc_single_tar, den_psc_multi_tar = [denoiser(arr) for arr in [psc_single_tar, psc_multi_tar]]

	# Configure priors
	beta_prior = 3 * np.ones(N)
	mu_prior = np.zeros(N)
	shape_prior_single, shape_prior_multi = np.ones(K_single), np.ones(K_multi)
	rate_prior_single, rate_prior_multi = 1e-1 * np.ones(K_single), 1e-1 * np.ones(K_multi)

	# Configure fit options
	iters = 75
	sigma = 1
	seed = 1
	y_xcorr_thresh = 1e-2
	max_penalty_iters = 50
	warm_start_lasso = True
	verbose = False
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

	priors = {
		'alpha': 0.15 * np.ones(N),
		'beta': 3e0 * np.ones(N),
		'mu': np.zeros(N),
		'phi': np.c_[0.125 * np.ones(N), 5 * np.ones(N)],
		'phi_cov': np.array([np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)]),
		'rate': sigma**2.,
		'shape': 1.
	}

	fit_options = { 
		'iters': iters,
		'num_mc_samples': 50,
		'y_xcorr_thresh': y_xcorr_thresh,
		'seed': seed,
		'phi_thresh_delay': -1,
		'learn_noise': True,
		'minimax_spk_prob': minimax_spike_prob,
		'scale_factor': scale_factor,
		'penalty': penalty,
		'lam_iters': 1,
		'noise_update': 'iid',
		'disc_strength': 0.,
		'minimum_spike_count': 3,
	}

	# Fit models
	model_single = adaprobe.Model(N, model_type='variational_sns', priors=priors)
	model_multi = adaprobe.Model(N, model_type='variational_sns', priors=priors)

	model_single.fit(den_psc_single_tar, stimulus_matrix_single_tar, fit_options=fit_options, method='cavi_sns')
	model_multi.fit(den_psc_multi_tar, stimulus_matrix_multi_tar, fit_options=fit_options, method='cavi_sns')

	d = {
		'model_multi_tar': model_multi,
		'model_single_tar': model_single,
		'stimulus_matrix_single_tar': stimulus_matrix_single_tar,
		'stimulus_matrix_multi_tar': stimulus_matrix_multi_tar,
		'psc_single_tar': psc_single_tar,
		'den_psc_single_tar': den_psc_single_tar,
		'psc_multi_tar': psc_multi_tar,
		'den_psc_multi_tar': den_psc_multi_tar,
		'img': f['img'],
		'targets': f['targets']
	}

	with bz2.BZ2File(args.data + '-CAVIaR_%s_%s.pkl'%(args.token, date.today().__str__()), 'wb') as savefile:
		cpickle.dump(d, savefile)