import numpy as np
from adaprobe import optimise
import time

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

		# Set up priors
		_ones = np.ones(self.n_presynaptic)

		if model_type == 'variational_sns':
			self.priors.setdefault('alpha', 1/2 * _ones)

		elif model_type == 'mbcs_multiplicative_noise':
			self.priors.setdefault('rho', 1e-1)

		self.priors.setdefault('shape', 1.)
		self.priors.setdefault('rate', 1.)
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
		"""Fit posterior distributions in offline mode.
		"""
		# assert method in ['mbcs', 'mbcs_sparse_outliers', 'cavi_sns']

		if method == 'mbcs':
			self._fit_mbcs(obs, stimuli, fit_options)
		elif method == 'mbcs_sparse_outliers':
			self._fit_mbcs_sparse_outliers(obs, stimuli, fit_options)
		elif method =='mbcs_multiplicative_noise':
			self._fit_mbcs_multiplicative_noise(obs, stimuli, fit_options)
		elif method =='mbcs_adaptive_threshold':
			self._fit_mbcs_adaptive_threshold(obs, stimuli, fit_options)
		elif method == 'cavi_sns':
			self._fit_cavi_sns(obs, stimuli, fit_options)
		else:
			raise Exception

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
