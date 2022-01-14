import numpy as np
from adaprobe import optimise
from scipy.stats import norm as normal
from adaprobe.utils import CrossValidation, sigmoid, sample_truncnorm, load_CV, load_CV_dir
from adaprobe.optimise.mbcs_spike_weighted_var import update_mu_constr_l1, update_isotonic_receptive_field, isotonic_filtering
from sklearn.linear_model import LinearRegression
import time
import os

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

class Ensemble:
	def __init__(self, ensemble):
		'''
		ensemble: list of Model class instances
		'''
		assert len(ensemble) > 1, 'Ensemble must consist of at least two models'
		self.ensemble = ensemble

	def merge(self, y, stim_matrix, minimum_spike_count=3, minimum_maximal_spike_prob=0.25, max_penalty_iters=50, max_lasso_iters=1000,
		constrain_weights=True, method='linear_regression', enforce_minimax=True):
		assert method in ['linear_regression', 'lasso']

		params = ['lam', 'mu', 'shape', 'rate']
		lam, mu, shape, rate = [np.mean(np.array([model.state[param] for model in self.ensemble]), axis=0) for param in params]
		
		if enforce_minimax:
			receptive_field, spike_prior = update_isotonic_receptive_field(lam, stim_matrix)
			mu, lam = isotonic_filtering(mu, lam, stim_matrix, receptive_field, minimum_spike_count=minimum_spike_count, 
				minimum_maximal_spike_prob=minimum_maximal_spike_prob)

		if method == 'linear_regression':
			mu = LinearRegression(positive=constrain_weights).fit(lam.T, y).coef_
		elif method == 'lasso':
			mu = update_mu_constr_l1(y, mu, lam, shape, rate)

		mu, lam = np.array(mu), np.array(lam) # convert from DeviceArray to ndarray array

		# Compute confidence in connections and appropriately rescale weights
		confidence = np.sum(np.array([mod.state['mu'] != 0 for mod in self.ensemble]), axis=0)/len(self.ensemble)
		mu *= confidence

		# Create new model 
		model = Model(mu.shape[0], priors=self.ensemble[0].priors, model_type=self.ensemble[0].model_type)

		for key, val in zip(params, [lam, mu, shape, rate]):
			model.state[key] = val

		model.state['connection_confidence'] = confidence

		return model

class Model:
	def __init__(self, N, model_type='mbcs', priors=dict()):
		"""Initialise adaprobe model.
		"""

		# assert model_type in ['mbcs', 'variational_sns',]

		# self.cell_locs = cell_locs
		self.n_presynaptic = N
		self.priors = priors
		self.trial_count = 0
		self.I = []
		self.L = []
		self.y = []
		self._cv = None
		self.model_type = model_type

		# Set up priors
		_ones = np.ones(self.n_presynaptic)

		if model_type == 'variational_sns':
			self.priors.setdefault('alpha', 1/2 * _ones)

		elif model_type == 'mbcs_multiplicative_noise':
			self.priors.setdefault('rho', 1e-1)

		# self.priors.setdefault('shape', 1.)
		# self.priors.setdefault('rate', 1.)
		self.priors.setdefault('sigma', np.ones(self.n_presynaptic))
		self.priors.setdefault('mu', np.zeros(self.n_presynaptic))
		self.priors.setdefault('beta', 1e1 * _ones)
		self.priors.setdefault('phi', np.c_[1e-1 * _ones, 5e0 * _ones])
		self.priors.setdefault('phi_cov', np.array([np.array([[1e-1, 0], [0, 1e0]]) 
			for _ in range(self.n_presynaptic)]))
		
		# Set initial state to prior
		self.reset()

	def reset(self):
		self.state = self.priors.copy()
		self.state['lam'] = []
		return

	'''
	Online updates no longer implemented

	def update(self, obs, stimuli, method='cavi_online_spike_and_slab', return_params=False, fit_options=dict()):
		"""Update posterior distributions given new (observation, stimulus) pair.
		"""
		if method == 'cavi_online_spike_and_slab':
			return self._update_cavi_online_spike_and_slab(obs, stimuli, return_params, fit_options)
		else:
			raise Exception("""[__Update exception__] Kwarg 'method' is invalid.""")
	'''

	def fit(self, obs, stimuli, method='mbcs', fit_options=dict()):
		'''Fit posterior distributions in offline mode.
		'''
		# assert method in ['mbcs', 'mbcs_sparse_outliers', 'cavi_sns']

		if method == 'mbcs':
			self._fit_mbcs(obs, stimuli, fit_options)
		elif method == 'mbcs_sparse_outliers':
			self._fit_mbcs_sparse_outliers(obs, stimuli, fit_options)
		elif method =='mbcs_multiplicative_noise':
			self._fit_mbcs_multiplicative_noise(obs, stimuli, fit_options)
		elif method =='mbcs_adaptive_threshold':
			self._fit_mbcs_adaptive_threshold(obs, stimuli, fit_options)
		elif method =='mbcs_cellwise_variance':
			self._fit_mbcs_cellwise_variance(obs, stimuli, fit_options)
		elif method == 'cavi_sns':
			self._fit_cavi_sns(obs, stimuli, fit_options)
		elif method == 'mbcs_spike_weighted_var':
			self._fit_mbcs_spike_weighted_var(obs, stimuli, fit_options)
		elif method == 'mbcs_spike_weighted_var_with_sigmoid_rf':
			self._fit_mbcs_spike_weighted_var_with_sigmoid_rf(obs, stimuli, fit_options)
		elif method == 'mbcs_spike_weighted_var_with_ghost':
			self._fit_mbcs_spike_weighted_var_with_ghost(obs, stimuli, fit_options)
		elif method == 'mbcs_spike_weighted_var_with_outliers':
			self._fit_mbcs_spike_weighted_var_with_outliers(obs, stimuli, fit_options)
		else:
			raise Exception

	def cross_validate(self, obs, stimuli, params, vals, method='mbcs', nfolds=10, fit_options=dict(), save_dir=None, token=None):
		'''Cross-validation.

		Args
			obs: 		postsynaptic responses
			stimuli: 	stimulus design matrix
			params: 	list parameters to be cross-validated
			vals: 		list of values associated with params

		'''

		# Initialise cross-validation record
		self._cv = CrossValidation(nfolds, params, vals)
		
		K = stimuli.shape[-1]
		random_order = np.random.choice(K, K, replace=False)
		split = np.array_split(random_order, nfolds)

		for idx in tqdm(range(nfolds), desc='CV fold'):

			# Load cross-validation data fold
			test_indices = split[idx]
			train_indices = np.setdiff1d(np.arange(K), split[idx])
			train_obs, train_stimuli = (obs[0][train_indices], obs[1][train_indices]), stimuli[:, train_indices]
			test_obs, test_stimuli = (obs[0][test_indices], obs[1][test_indices]), stimuli[:, test_indices]

			# Revert to initial priors and re-fit model
			self.reset() 
			self.fit(train_obs, train_stimuli, method=method, fit_options=fit_options)
			lppd, ppd_samples = self.eval_posterior_predictive_density(test_obs[0], test_stimuli, method)
			self._cv.update(fold=idx, test_obs=test_obs[0], test_stim=test_stimuli,
			 predictive_distribution=ppd_samples, lppd=lppd)

		print(self._cv)
		
		if save_dir is not None:
			print('Saving cross-validation object to file...')

			if save_dir[-1] != '/': save_dir += '/'
			if not os.path.isdir(save_dir):
				os.mkdir(save_dir)

			if token[-1] != '_': token += '_'
			self._cv.save(save_dir + token)

		print('Cross-validation complete.')

	def model_selection(self, fdir):
		'''Select model from supplied directory via mean lppd.
		'''
		return load_CV_dir(fdir)

	def eval_posterior_predictive_density(self, obs, stimuli, method, n_samples=100, epsilon=1e-7):
		'''Evaluate log pointwise predictive density (see Gelman et al. (2014), CRC Press, pp. 168-169)

			obs 	: K x 1 array of observed PSCs
			stimuli : N x K matrix of power delivered to target neurons

		'''
		assert method in ['mbcs', 'mbcs_adaptive_threshold']

		# Sample from posterior
		w = np.random.normal(self.state['mu'], self.state['beta'] * (self.state['mu'] != 0), 
			[n_samples, self.n_presynaptic])
		phi = np.array([sample_truncnorm(self.state['phi'][n], np.diag(self.state['phi_cov'][n]),
			size=[n_samples, 1]) for n in range(self.n_presynaptic)]).astype(float)
		s = np.array([np.random.rand(self.n_presynaptic, stimuli.shape[-1]) <= \
			sigmoid(phi[:, j, 0][:, None] * stimuli - phi[:, j, 1][:, None]) for j in range(n_samples)]).astype(float)
		sig = np.sqrt(self.state['rate']/self.state['shape'])

		# Reconstruct obs
		y_pred = np.sum(w[..., None] * s, 1)

		# Compute lppd
		lppd = np.sum(np.log(np.mean(normal.pdf(obs, y_pred, sig), axis=0) + epsilon))

		return lppd, y_pred

	def _fit_cavi_sns(self, obs, stimuli, fit_options):
		"""Run CAVI with spike-and-slab synapse prior.
		"""
		t_start = time.time()

		result = optimise.cavi_sns(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['shape'], 
			self.state['rate'], self.state['phi'], self.state['phi_cov'], **fit_options 
		)
		t_end = time.time()

		mu, beta, alpha, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		alpha 		= np.array(alpha)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		alpha_hist 		= np.array(alpha_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)		

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['alpha']		= alpha
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam.T
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'alpha': alpha_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist
		}

		return

	def _fit_mbcs(self, obs, stimuli, fit_options):
		"""Run MBCS with .
		"""
		t_start = time.time()
		result = optimise.mbcs(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist
		}

		return

	def _fit_mbcs_sparse_outliers(self, obs, stimuli, fit_options):
		"""Run MBCS with sparse outliers.
		"""
		t_start = time.time()
		result = optimise.mbcs_sparse_outliers(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, z, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)
		z 			= np.array(z)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)
		z_hist 			= np.array(z_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam.T
		self.state['z'] 		= z
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
			'z_hist': z_hist
		}

		return

	def _fit_mbcs_multiplicative_noise(self, obs, stimuli, fit_options):
		"""
		"""
		t_start = time.time()
		result = optimise.mbcs_multiplicative_noise(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], self.state['rho'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, xi, rho, z = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)
		xi 			= np.array(xi)
		rho 		= np.array(rho)
		z 			= np.array(z)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam.T
		self.state['xi'] 		= xi
		self.state['rho']		= rho
		self.state['z'] 		= z
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		return

	def _fit_mbcs_adaptive_threshold(self, obs, stimuli, fit_options):
		"""
		"""
		t_start = time.time()
		result = optimise.mbcs_adaptive_threshold(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, z, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)
		z 			= np.array(z)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)
		z_hist 			= np.array(z_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.state['z'] 		= z
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
			'z': z_hist
		}

	def _fit_mbcs_cellwise_variance(self, obs, stimuli, fit_options):
		"""
		"""
		t_start = time.time()
		result = optimise.mbcs_cellwise_variance(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['sigma'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, sigma, phi, phi_cov, z, mu_hist, beta_hist, lam_hist, sigma_hist, \
		phi_hist, phi_cov_hist, z_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		sigma 		= np.array(sigma)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)
		z 			= np.array(z)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		sigma_hist 		= np.array(sigma_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)
		z_hist 			= np.array(z_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['sigma'] 	= sigma
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.state['z'] 		= z
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'sigma': sigma_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
			'z': z_hist
		}

	def _fit_mbcs_spike_weighted_var(self, obs, stimuli, fit_options):
		'''
		'''
		t_start = time.time()
		result = optimise.mbcs_spike_weighted_var(
			obs, stimuli, self.state['mu'], self.state['beta'],	self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, \
		rate_hist, phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
		}

	def _fit_mbcs_spike_weighted_var_with_sigmoid_rf(self, obs, stimuli, fit_options):
		'''
		'''
		t_start = time.time()
		result = optimise.mbcs_spike_weighted_var_with_sigmoid_rf(
			obs, stimuli, self.state['mu'], self.state['beta'],	self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, \
		rate_hist, phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
		}

	def _fit_mbcs_spike_weighted_var_with_ghost(self, obs, stimuli, fit_options):
		'''
		'''
		t_start = time.time()
		result = optimise.mbcs_spike_weighted_var_with_ghost(
			obs, stimuli, self.state['mu'], self.state['beta'],	self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, \
		rate_hist, phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
		}

	def _fit_mbcs_spike_weighted_var_with_outliers(self, obs, stimuli, fit_options):
		t_start = time.time()
		result = optimise.mbcs_spike_weighted_var_with_outliers(
			obs, stimuli, self.state['mu'], self.state['beta'],	self.state['shape'], self.state['rate'], 
			self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, z, mu_hist, beta_hist, lam_hist, shape_hist, \
		rate_hist, phi_hist, phi_cov_hist, z_hist = result

		# move from GPU back to CPU
		## param vectors
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)
		z 			= np.array(z)

		## history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)
		z_hist 			= np.array(z_hist)

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= lam
		self.state['z'] 		= z
		self.trial_count 		= lam.shape[1]
		self.time 				= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi': phi_hist,
			'phi_cov': phi_cov_hist,
			'z': z_hist
		}