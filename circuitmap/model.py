import numpy as np
import time
import os
from copy import deepcopy

from circuitmap import optimise

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

class Model:
	def __init__(self, N, priors=dict()):
		''' Initialise circuitmap model.
		'''

		self.N = N
		self.priors = priors

		# Set up priors
		self.priors.setdefault('alpha', 1/4 * np.ones(N))
		self.priors.setdefault('phi', np.c_[1e-1 * np.ones(N), 5e0 * np.ones(N)])
		self.priors.setdefault('phi_cov', np.array([np.array([[1e-1, 0], [0, 1e0]]) 
			for _ in range(N)]))
		self.priors.setdefault('mu', np.zeros(N))
		self.priors.setdefault('beta', 1e1 * np.ones(N))
		self.priors.setdefault('shape', 1.)
		self.priors.setdefault('rate', 1e-1)
		
		self.state = deepcopy(self.priors)
		
		# self.state = self.priors.copy()

		# Set initial state to prior
		# self.reset()

	# def reset(self):
	# 	self.state = self.priors.copy()
	# 	return

	def fit(self, obs, stimuli, method='caviar', fit_options=dict()):
		if method == 'mbcs':
			self._fit_mbcs(obs, stimuli, fit_options)
		elif method == 'cavi_sns':
			self._fit_cavi_sns(obs, stimuli, fit_options)
		elif method == 'caviar':
			self._fit_caviar(obs, stimuli, fit_options)
		else:
			raise Exception

	def _fit_cavi_sns(self, obs, stimuli, fit_options):
		t_start = time.time()

		result = optimise.cavi_sns(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['shape'], 
			self.state['rate'], self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, alpha, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist = result

		# move from GPU back to CPU
		mu 			= np.array(mu)
		beta 		= np.array(beta)
		alpha 		= np.array(alpha)
		lam 		= np.array(lam)
		shape 		= np.array(shape)
		rate 		= np.array(rate)
		phi 		= np.array(phi)
		phi_cov 	= np.array(phi_cov)

		# history vectors
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
		self.state['lam'] 		= lam
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
			'phi_cov': phi_cov_hist,
		}

		return

	def _fit_caviar(self, obs, stimuli, fit_options):
		t_start = time.time()

		result = optimise.caviar(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], 
			self.state['rate'], self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, phi, phi_cov, z, receptive_fields, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist = result

		# move from GPU back to CPU
		mu 					= np.array(mu)
		beta 				= np.array(beta)
		lam 				= np.array(lam)
		shape 				= np.array(shape)
		rate 				= np.array(rate)
		phi 				= np.array(phi)
		phi_cov 			= np.array(phi_cov)
		z 					= np.array(z)
		receptive_fields 	= np.array(receptive_fields)

		# history vectors
		mu_hist 		= np.array(mu_hist)
		beta_hist 		= np.array(beta_hist)
		lam_hist 		= np.array(lam_hist)
		shape_hist 		= np.array(shape_hist)
		rate_hist 		= np.array(rate_hist)
		phi_hist 		= np.array(phi_hist)
		phi_cov_hist 	= np.array(phi_cov_hist)
		z_hist			= np.array(z_hist)

		self.state['mu'] 				= mu
		self.state['beta'] 				= beta
		self.state['shape'] 			= shape
		self.state['rate'] 				= rate
		self.state['phi'] 				= phi
		self.state['phi_cov'] 			= phi_cov
		self.state['lam'] 				= lam
		self.state['z'] 				= z
		self.state['receptive_fields'] 	= receptive_fields
		self.trial_count 				= lam.shape[1]
		self.time 						= t_end - t_start

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

		return

	def _fit_mbcs(self, obs, stimuli, fit_options):
		t_start = time.time()

		result = optimise.mbcs(
			obs, stimuli, self.state['mu'], self.state['beta'],	self.state['shape'], self.state['rate'], 
			**fit_options
		)

		t_end = time.time()

		mu, beta, lam, shape, rate, z, receptive_fields, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, z_hist = result

		# move from GPU back to CPU
		mu 					= np.array(mu)
		beta 				= np.array(beta)
		lam 				= np.array(lam)
		shape 				= np.array(shape)
		rate 				= np.array(rate)
		z 					= np.array(z)
		receptive_fields 	= np.array(receptive_fields)

		# history vectors
		mu_hist		= np.array(mu_hist)
		beta_hist 	= np.array(beta_hist)
		lam_hist 	= np.array(lam_hist)
		shape_hist 	= np.array(shape_hist)
		rate_hist 	= np.array(rate_hist)
		z_hist 		= np.array(z_hist)

		self.state['mu'] 				= mu
		self.state['beta'] 				= beta
		self.state['shape'] 			= shape
		self.state['rate'] 				= rate
		self.state['lam'] 				= lam
		self.state['z'] 				= z
		self.state['receptive_fields'] 	= receptive_fields
		self.trial_count 				= lam.shape[1]
		self.time 						= t_end - t_start

		# Set up history dict
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'z': z_hist
		}