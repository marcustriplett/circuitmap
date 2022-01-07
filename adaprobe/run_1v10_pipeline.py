import argparse
import numpy as np
import adaprobe
from adaprobe.psc_denoiser import NeuralDenoiser
import _pickle as cpickle # pickle compression
import bz2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--ensemble_size')
	parser.add_argument('--denoiser')
	parser.add_argument('--iters')
	args = parser.parse_args()

	# Load data
	f = np.load(args.data)
	psc_single_tar, psc_multi_tar = [f[param] for param in ['psc_single_tar', 'psc_multi_tar']]
	stimulus_matrix_single_tar, stimulus_matrix_multi_tar = [f[param] for param in ['stimulus_matrix_single_tar', 'stimulus_matrix_multi_tar']]
	N, K_single = stimulus_matrix_single_tar.shape
	K_multi = stimulus_matrix_multi_tar.shape[-1]

	# Denoise traces
	denoiser = NeuralDenoiser(path=args.denoiser)
	den_psc_single_tar, den_psc_multi_tar = [denoiser(arr) for arr in [psc_single_tar, psc_multi_tar]]
	y_single, y_multi = [np.trapz(arr, axis=1) for arr in [den_psc_single_tar, den_psc_multi_tar]]

	# Configure priors
	beta_prior = 3 * np.ones(N)
	mu_prior = np.zeros(N)
	shape_prior_single, shape_prior_multi = np.ones(K_single), np.ones(K_multi)
	rate_prior_single, rate_prior_multi = 1e-1 * np.ones(K_single), 1e-1 * np.ones(K_multi)

	# Configure fit options
	iters = np.float(args.iters)
	seed = 1
	y_xcorr_thresh = 1e-2
	max_penalty_iters = 50
	warm_start_lasso = True
	verbose = False
	minimum_spike_count = 3
	num_mc_samples_noise_model = 100
	minimum_maximal_spike_prob = 0.25
	noise_scale = 0.5
	init_spike_prior = 0.5
	spont_rate = 0.2
	num_mc_samples = 500
	penalty = 2
	max_lasso_iters = 1000
	scale_factor = 0.75
	constrain_weights = 'positive'
	lam_masking = True


	priors_single = {
		'beta': beta_prior,
		'mu': mu_prior,
		'shape': shape_prior_single,
		'rate': rate_prior_single
	}

	priors_multi = {
		'beta': beta_prior,
		'mu': mu_prior,
		'shape': shape_prior_multi,
		'rate': rate_prior_multi
	}

	fit_options = { 
		'iters': iters,
		'num_mc_samples': num_mc_samples,
		'penalty': penalty,
		'max_penalty_iters': max_penalty_iters,
		'max_lasso_iters': max_lasso_iters,
		'scale_factor': scale_factor,
		'constrain_weights': constrain_weights,
		'lam_masking': lam_masking,
		'y_xcorr_thresh': y_xcorr_thresh,
		'seed': seed,
		'verbose': verbose,
		'warm_start_lasso': warm_start_lasso,
		'minimum_spike_count': minimum_spike_count,
		'minimum_maximal_spike_prob': minimum_maximal_spike_prob,
		'noise_scale': noise_scale,
		'init_spike_prior': init_spike_prior,
		'spont_rate': spont_rate
	}


	# Fit models
	model_single = adaprobe.Model(N, model_type='mbcs', priors=priors_single)
	models_multi = [adaprobe.Model(N, model_type='mbcs', priors=priors_multi) for _ in range(args.ensemble_size)]

	model_single.fit((y_single, den_psc_single_tar), stimulus_matrix_single_tar, fit_options=fit_options, method='mbcs_spike_weighted_var')
	for model in models_multi:
		model.fit((y_multi, den_psc_multi_tar), stimulus_matrix_multi_tar, fit_options=fit_options, method='mbcs_spike_weighted_var')

	# Merge ensemble
	ensemble_model = adaprobe.Ensemble(models_multi).merge(y_multi, stimulus_matrix_multi_tar)

	d = {
		'ensemble_model_multi_tar': ensemble_model,
		'ensemble_multi_tar': models_multi,
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

	with bz2.BZ2File(args.data + '-analysis.pkl', 'wb') as savefile:
		cpickle.dump(d, savefile)

# sbatch run_pipeline.sh path/to/folder 10 