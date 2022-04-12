import numpy as np
from tqdm import tqdm
import argparse

# adaprobe imports
import adaprobe
from adaprobe.simulation import simulate_continuous_experiment
from adaprobe.psc_denoiser import NeuralDenoiser
from adaprobe.optimise.cosamp import cosamp

import pandas as pd
import yaml
import sys
import os

if __name__ == '__main__':
	# config GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'
	os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

	# non-YAML settings
	sampling_freq = 20000
	ground_truth_eval_batch_size = 500

	# read input args
	parser = argparse.ArgumentParser()
	parser.add_argument('--config')
	parser.add_argument('--save_dir')
	parser.add_argument('--token')
	args = parser.parse_args()

	config_path = args.config
	save_dir = args.save_dir
	token = args.token

	# load config yaml
	config = yaml.safe_load(open(config_path))

	N 							= config['N']
	Hs 							= config['Hs']
	stim_freqs 					= config['stim_freqs']
	expt_len 					= config['expt_len'] * sampling_freq # assumes dict element in seconds
	subsample_every 			= config['subsample_every'] * sampling_freq # assumes dict element in seconds
	connection_prob 			= config['connection_prob']
	spont_rate 					= config['spont_rate'] # Hz
	max_power_min_spike_rate 	= config['max_power_min_spike_rate']
	denoiser 					= NeuralDenoiser(path=config['denoiser'])

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

	priors_mbcs = {
		'beta': beta_prior,
		'mu': mu_prior,
	}

	# config fit options
	iters = 50
	seed = 1
	y_xcorr_thresh = 1e-2
	max_penalty_iters = 50
	warm_start_lasso = True
	minimum_spike_count = 3
	num_mc_samples_noise_model = 100
	minimax_spike_prob = max_power_min_spike_rate
	noise_scale = 0.5
	init_spike_prior = 0.95
	num_mc_samples = 500
	max_lasso_iters = 1000
	scale_factor = 0.75
	constrain_weights = 'positive'
	orthogonal_outliers = True
	lam_mask_fraction = 0.025
	outlier_penalty = 1e1
	verbose = False
	delay_spont_estimation = -1
	penalty = 1e0
	noise_update = 'iid'

	fit_options_caviar = { 
		'iters': iters,
		'num_mc_samples': 50,
		'y_xcorr_thresh': y_xcorr_thresh,
		'seed': seed,
		'minimax_spk_prob': minimax_spike_prob,
		'scale_factor': scale_factor,
		'penalty': penalty,
		'minimum_spike_count': minimum_spike_count,
	}

	fit_options_mbcs = { 
		'iters': iters,
		'num_mc_samples': num_mc_samples,
		'penalty': penalty,
		'max_penalty_iters': max_penalty_iters,
		'max_lasso_iters': max_lasso_iters,
		'scale_factor': scale_factor,
		'constrain_weights': constrain_weights,
		'y_xcorr_thresh': y_xcorr_thresh,
		'seed': seed,
		'verbose': verbose,
		'warm_start_lasso': warm_start_lasso,
		'minimum_spike_count': minimum_spike_count,
		'minimum_maximal_spike_prob': minimax_spike_prob,
		'noise_scale': noise_scale,
		'init_spike_prior': init_spike_prior,
		'orthogonal_outliers': orthogonal_outliers,
		'lam_mask_fraction': lam_mask_fraction,
		'outlier_penalty': outlier_penalty,
		'delay_spont_estimation': delay_spont_estimation,
	}

	# init results dataframe
	header = ['N', 'H', 'stim_freq', 'NWD', 'weights', 'tstep', 'CAVIaR', 'SnS', 'CoSaMP', 'MBCS', 'CAVIaR_t', 'SnS_t', 'CoSaMP_t', 'MBCS_t']
	results = pd.DataFrame(columns=header)
	nwd_status = [False, True]

	# init weights to None
	weights = None

	for (sf, H) in itertools.product(stim_freqs, Hs):
		print('Simulating mapping experiment with %i targets at %i Hz'%(H, sf))

		# simulate experiment
		expt = simulate_continuous_experiment(N=N, H=H, expt_len=expt_len, stim_freq=sf, 
			connection_prob=connection_prob, spont_rate=spont_rate, weights=weights, 
			ground_truth_eval_batch_size=ground_truth_eval_batch_size, 
			max_power_min_spike_rate=max_power_min_spike_rate)

		weights = expt['weights'] # set weights from first run

		psc = expt['obs_responses']
		den_psc = denoiser(psc)
		true_resp = expt['true_responses']
		stim_matrix = expt['stim_matrix']
		n_connected = np.sum(weights != 0)

		tsteps = np.arange(subsample_every, expt_len + 1, subsample_every)
		tsteps_sec = list(map(str, (tsteps/sampling_freq).astype(int))) # for pd dataframe

		for t, subsample_len in enumerate(tsteps):
			subsample_trials = int(subsample_len/isi)
			psc_subsample = psc[:subsample_len]
			den_psc_subsample = den_psc[:subsample_len]
			stim_matrix_subsample = stim_matrix[:, :subsample_len]

			# MBCS noise priors depend on trial length
			priors_mbcs['shape'] = np.ones(subsample_trials)
			priors_mbcs['rate'] = 1e-1 * np.ones(subsample_trials)

			models_caviar = [adaprobe.Model(model_type='variational_sns', priors=priors_caviar) for _ in range(2)]
			models_sns = [adaprobe.Model(model_type='variational_sns', priors=priors_caviar) for _ in range(2)]
			models_mbcs = [adaprobe.Model(model_type='mbcs', priors=priors_mbcs) for _ in range(2)]
			models_cosamp = [None for _ in range(2)]
			models_cosamp_t = [None for _ in range(2)]

			for model, data in zip(models_caviar, [psc_subsample, den_psc_subsample]):
				model.fit(data, stim_matrix_subsample, fit_options=fit_options_caviar, method='caviar')

			for model, data in zip(models_sns, [psc_subsample, den_psc_subsample]):
				model.fit(data, stim_matrix_subsample, fit_options=fit_options_caviar, method='cavi_sns')

			for model, data in zip(models_mbcs, [psc_subsample, den_psc_subsample]):
				model.fit(data, stim_matrix_subsample, fit_options=fit_options_mbcs, method='mbcs_spike_weighted_var_with_outliers')

			for model, model_t, data in zip(models_cosamp, models_cosamp_t, [psc_subsample, den_psc_subsample]):
				cos = cosamp((stim_matrix_subsample != 0).astype(float).T, np.trapz(data, axis=-1), n_connected)
				model = cos[0]
				model_t = cos[1]

			for mod_indx in range(2):
				results.append({
					'N': N,
					'H': H,
					'stim_freq': sf,
					'NWD': nwd_status[mod_indx],
					'weights': weights,
					'tstep': tsteps_sec[t],
					'CAVIaR': models_caviar[mod_indx].state['mu'],
					'SnS': models_sns[mod_indx].state['mu'] * models_sns[mod_indx].state['alpha'],
					'MBCS': models_mbcs[mod_indx].state['mu'],
					'CoSaMP': models_cosamp[mod_indx],
					'CAVIaR_t': models_caviar[mod_indx].time,
					'SnS_t': models_sns[mod_indx].time,
					'MBCS_t': models_mbcs[mod_indx].time,
					'CoSaMP_t': models_cosamp_t[mod_indx]
				}, ignore_index=True)

	if save_dir[-1] != '/': save_dir += '/'

	if token is None:
		token = ''
	elif token[0] != '_':
		token = '_' + token

	results.to_json(save_dir \
		+ 'N%i_connprob%.2f_spontrate%.2f_minspikerate%.2f_exptlen%i'%(N, connection_prob, spont_rate, max_power_min_spike_rate, config['expt_len'])\
		+ token)
