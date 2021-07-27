import numpy as np
from adaprobe import optimise

class Model:
	def __init__(self, cell_locs, variational_model='factorised', priors=dict()):
		"""Initialise adaprobe model.

		Args:
			cell_locs: 
			priors:

		"""

		self.cell_locs = cell_locs
		self.n_presynaptic = cell_locs.shape[0]
		self.priors = priors
		self.trial_count = 0
		self.I = []
		self.L = []
		self.y = []

		# Set up priors
		_ones = np.ones(self.n_presynaptic)

		self.priors.setdefault('alpha', 1/2 * _ones)
		self.priors.setdefault('shape', 1.)
		self.priors.setdefault('rate', 1.)
		self.priors.setdefault('mu', np.zeros(self.n_presynaptic))
		self.priors.setdefault('beta', 3 * _ones)
		self.priors.setdefault('phi', np.c_[1e-1 * _ones, 1e1 * _ones])
		self.priors.setdefault('phi_cov', np.array([np.array([[1e-1, 0], [0, 5e0]]) 
			for _ in range(self.n_presynaptic)]))
		
		# Set initial state to prior
		self.reset()

	def reset(self):
		self.state = self.priors.copy()
		self.state['lam'] = []
		return

	def update(self, obs, stimuli, method='cavi_online_spike_and_slab', return_params=False, fit_options=dict()):
		"""Update posterior distributions given new (observation, stimulus) pair.
		"""
		if method == 'cavi_online_spike_and_slab':
			return self._update_cavi_online_spike_and_slab(obs, stimuli, return_params, fit_options)
		else:
			raise Exception("""[__Update exception__] Kwarg 'method' is invalid.""")

	def fit(self, obs, stimuli, fit_options=dict()):
		"""Fit posterior distributions in offline mode.
		"""
		self._fit_cavi_offline_spike_and_slab_NOTS_jax(obs, stimuli, fit_options)

	def _fit_cavi_offline_spike_and_slab_NOTS_jax(self, obs, stimuli, fit_options):
		"""Run CAVI in offline mode with spike-and-slab synapse prior.
		"""
		result = optimise.cavi_offline_spike_and_slab_NOTS_jax(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['shape'], 
			self.state['rate'], self.state['phi'], self.state['phi_cov'], **fit_options 
		)

		mu, beta, alpha, lam, shape, rate, phi, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist = result

		# mu, beta, alpha, lam, shape, rate, phi, phi_cov = result

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['alpha']		= alpha
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi'] 		= phi
		self.state['phi_0'] 	= phi[:, 0]
		self.state['phi_1'] 	= phi[:, 1]
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= list(lam.T)
		self.trial_count 		= obs.shape[0]

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
