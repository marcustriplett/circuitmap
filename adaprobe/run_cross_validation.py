import os
import sys
sys.path.append('..')
import adaprobe
from adaprobe.psc_denoiser import NeuralDenoiser
import numpy as np
import argparse

if __name__ == '__main__':
	'''Cross-validation
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'

	# Process inputs
	parser = argparse.ArgumentParser()
	args = ['data_dir', 'filename', 'sigma', 'seed', 'save_dir', 'nfolds', 'denoiser']
	for arg in args:		
		parser.add_argument('--' + arg)
	args = parser.parse_args()

	# Load sigma and seed
	data_dir = args.data_dir
	filename = args.filename
	sigma = np.float(args.sigma)
	seed = np.int(args.seed)
	save_dir = args.save_dir
	nfolds = np.int(args.nfolds)
	denoiser_path = args.denoiser

	# Load data and NN denoiser
	for path in [data_dir, save_dir]:
		if path[-1] != '/': path += '/'
	data = np.load(data_dir + filename)
	denoiser = NeuralDenoiser(path=denoiser_path)
	psc_seq, psc_multi = data['psc_seq'], data['psc_multi']
	stim_matrix_seq, stim_matrix_multi = data['stimulus_matrix_seq'], data['stimulus_matrix_multi']
	N, K = stim_matrix_seq.shape

	# Denoise traces
	seq_max, multi_max = [np.max(arr, axis=1)[:, None] for arr in [psc_seq, psc_multi]]
	den_psc_seq = denoiser(psc_seq)
	den_psc_multi = denoiser(psc_multi)

	# Configure priors
	phi_prior = np.c_[0.125 * np.ones(N), 5 * np.ones(N)]
	phi_cov_prior = np.array([np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])
	alpha_prior = 0.15 * np.ones(N)
	beta_prior = 3e0 * np.ones(N)
	mu_prior = np.zeros(N)

	# Configure optimisation params
	phi_thresh = 0.09
	phi_delay = -1
	y_xcorr_thresh = 1e-3
	outlier_penalty = 30
	orthogonal_outliers = True
	learn_lam = True
	lam_masking = True
	constrain_weights = 'positive'

	priors = {
		'beta': beta_prior,
		'mu': mu_prior,
		'phi': phi_prior,
		'phi_cov': phi_cov_prior,
		'shape': 1,
		'rate': sigma**2
	}

	fit_options = { 
		'iters': 50,
		'num_mc_samples': 500,
		'penalty': 2,
		'max_penalty_iters': 20, # default 15
		'max_lasso_iters': 1000,
		'scale_factor': 0.75,
		'constrain_weights': constrain_weights,
		'lam_masking': lam_masking,
		'y_xcorr_thresh': y_xcorr_thresh,
		'learn_lam': learn_lam,
		'outlier_penalty': outlier_penalty,
		'phi_thresh': phi_thresh,
		'seed': seed,
		'orthogonal_outliers': orthogonal_outliers,
		'phi_delay': phi_delay
	}

	models = [adaprobe.Model(N, model_type='mbcs', priors=priors) for _ in range(2)]
	datasets = [den_psc_seq, den_psc_multi]
	stimuli = [stim_matrix_seq, stim_matrix_multi]
	labels = ['single', 'multi']

	for model, traces, stim, label in zip(models, datasets, stimuli, labels):
		model.cross_validate((np.trapz(traces, axis=1), traces), stim, method='mbcs_adaptive_threshold',
		 nfolds=nfolds, fit_options=fit_options, save_dir=save_dir, token=filename[:-4]+'_'+label+'_')



