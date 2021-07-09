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

	def update(self, obs, stimuli, method='cavi_online_spike_and_slab', 
		return_params=False, fit_options=dict()):
		"""Update posterior distributions given new (observation, stimulus) pair.
		"""
		if method == 'cavi_online_spike_and_slab':
			return self._update_cavi_online_spike_and_slab(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_joint_vi':
			return self._update_cavi_online_spike_and_slab_joint_vi(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_multistim':
			return self._update_cavi_online_spike_and_slab_multistim(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_no_ots':
			return self._update_cavi_online_spike_and_slab_no_ots(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_spike_noise':
			return self._update_cavi_online_spike_and_slab_spike_noise(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_3d_omega':
			return self._update_cavi_online_spike_and_slab_3d_omega(obs, stimuli, return_params, fit_options)
		elif method == 'cavi_online_spike_and_slab_3d_laplace':
			return self._update_cavi_online_spike_and_slab_3d_laplace(obs, stimuli, return_params, fit_options)
		else:
			raise Exception("""[__Update exception__] Kwarg 'method' is invalid.""")

	def fit(self, obs, stimuli, method='cavi', fit_options=dict()):
		"""Fit posterior distributions in offline mode.
		"""
		if method == 'cavi':
			self._fit_cavi(obs, stimuli, fit_options)
		elif method == 'cavi_spike_and_slab':
			self._fit_cavi_spike_and_slab(obs, stimuli, fit_options)
		elif method == 'cavi_offline_spike_and_slab_3d_omega':
			self._fit_cavi_offline_spike_and_slab_3d_omega(obs, stimuli, fit_options)
		elif method == 'cavi_offline_spike_and_slab_NOTS_jax':
			self._fit_cavi_offline_spike_and_slab_NOTS_jax(obs, stimuli, fit_options)
		else:
			raise Exception


	def _update_cavi_online(self, obs, stimuli, return_params, fit_options):
		"""
			DEPRECATED

			Run CAVI in online mode without spike-and-slab prior
		"""

		mu, beta, lam, shape, rate, phi_map, phi_cov = optimise.cavi_online(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], 
			self.state['rate'], self.state['phi_map'], self.state['phi_cov'], 
			self.state['omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, lam, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov

			self.state['lam'].append(lam)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _update_cavi_online_spike_and_slab(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with spike-and-slab prior.
		"""
		mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov = optimise.cavi_online_spike_and_slab(
			obs, stimuli, self.state['mu'], self.state['beta'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _update_cavi_online_spike_and_slab_multistim(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with spike-and-slab prior.
		"""
		mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov = optimise.cavi_online_spike_and_slab_multistim(
			obs, stimuli, self.state['mu'], self.state['beta'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _update_cavi_online_spike_and_slab_joint_vi(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with non-factorised spike-and-slab/excitability prior.
		"""
		mu, Lam, alpha, lamk, shape, rate = optimise.cavi_online_spike_and_slab_joint_vi(
			obs, stimuli, self.state['mu'], self.state['Lam'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['omega'], 
			self.cell_locs, **fit_options
		)

		if return_params:
			return mu, Lam, alpha, lamk, shape, rate
		else:
			self.state['mu'] 		= mu
			self.state['Lam'] 		= Lam
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])


	def _update_cavi_online_spike_and_slab_no_ots(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with non-factorised spike-and-slab/excitability prior.
		"""
		mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov = optimise.cavi_online_spike_and_slab_no_ots(
			obs, stimuli, self.state['mu'], self.state['beta'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])


	def _update_cavi_online_spike_and_slab_spike_noise(self, obs, stimuli, return_params, fit_options=dict()):
		mu, beta, alpha, lamk, zetak, etak, phi_map, phi_cov = optimise.cavi_online_spike_and_slab_spike_noise(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['omega'], self.state['sigma'], self.state['rho'], self.cell_locs, 
			**fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, zetak, etak, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.state['zeta'].append(zetak)
			self.state['eta'].append(etak)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _update_cavi_online_spike_and_slab_3d_omega(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with spike-and-slab prior and 3d Omega.
		"""
		mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov = optimise.cavi_online_spike_and_slab_3d_omega(
			obs, stimuli, self.state['mu'], self.state['beta'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['Omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _update_cavi_online_spike_and_slab_3d_laplace(self, obs, stimuli, return_params, fit_options=dict()):
		"""Run CAVI in online mode with spike-and-slab prior and 3d Omega.
		"""
		mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov = optimise.cavi_online_spike_and_slab_3d_laplace(
			obs, stimuli, self.state['mu'], self.state['beta'], 
			self.state['alpha'], self.state['shape'], self.state['rate'], self.state['phi_map'], 
			self.state['phi_cov'], self.state['omega'], self.cell_locs, **fit_options
		)

		if return_params:
			return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
		else:
			self.state['mu'] 		= mu
			self.state['beta'] 		= beta
			self.state['alpha'] 	= alpha
			self.state['shape'] 	= shape
			self.state['rate'] 		= rate
			self.state['phi_map'] 	= phi_map
			self.state['phi_0'] 	= phi_map[:, 0]
			self.state['phi_1'] 	= phi_map[:, 1]
			self.state['phi_cov'] 	= phi_cov
			
			self.state['lam'].append(lamk)
			self.trial_count += 1

			# record stimuli
			self.L.append(stimuli[0])
			self.I.append(stimuli[1])

	def _fit_cavi(self, obs, stimuli, fit_options):
		"""Run CAVI in offline mode with Gaussian synapse prior.
		"""
		result = optimise.cavi_offline(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['shape'], self.state['rate'],
			self.state['phi_map'], self.state['phi_cov'], self.state['omega'], self.cell_locs, **fit_options 
		)

		mu, beta, lam, shape, rate, phi_map, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_map_hist, phi_cov_hist = result

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi_map'] 	= phi_map
		self.state['phi_0'] 	= phi_map[:, 0]
		self.state['phi_1'] 	= phi_map[:, 1]
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= list(lam.T)
		self.trial_count 		= obs.shape[0]

		# Set up history dict.
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi_map': phi_map_hist,
			'phi_cov': phi_cov_hist
		}


	def _fit_cavi_spike_and_slab(self, obs, stimuli, fit_options):
		"""Run CAVI in offline mode with spike-and-slab synapse prior.
		"""
		result = optimise.cavi_offline_spike_and_slab(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['shape'], 
			self.state['rate'], self.state['phi_map'], self.state['phi_cov'], self.state['omega'], 
			self.cell_locs, **fit_options 
		)

		mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_map_hist, phi_cov_hist = result

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['alpha']		= alpha
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi_map'] 	= phi_map
		self.state['phi_0'] 	= phi_map[:, 0]
		self.state['phi_1'] 	= phi_map[:, 1]
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= list(lam.T)
		self.trial_count 		= obs.shape[0]

		# Set up history dict.
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'alpha': alpha_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi_map': phi_map_hist,
			'phi_cov': phi_cov_hist
		}

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

	def _fit_cavi_offline_spike_and_slab_3d_omega(self, obs, stimuli, fit_options):
		"""Run CAVI in offline mode in three dimensions with a spike-and-slab synapse prior.
		"""

		result = optimise.cavi_offline_spike_and_slab_3d_omega(
			obs, stimuli, self.priors['mu'], self.priors['beta'], self.priors['alpha'], self.priors['shape'], 
			self.priors['rate'], self.priors['phi_map'], self.priors['phi_cov'], self.state['Omega'], 
			self.cell_locs, **fit_options
		)

		mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_map_hist, phi_cov_hist = result

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['alpha']		= alpha
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['phi_map'] 	= phi_map
		self.state['phi_0'] 	= phi_map[:, 0]
		self.state['phi_1'] 	= phi_map[:, 1]
		self.state['phi_cov'] 	= phi_cov
		self.state['lam'] 		= list(lam.T)
		self.trial_count 		= obs.shape[0]

		# Set up history dict.
		self.history = {
			'mu': mu_hist,
			'beta': beta_hist,
			'alpha': alpha_hist,
			'lam': lam_hist,
			'shape': shape_hist,
			'rate': rate_hist,
			'phi_map': phi_map_hist,
			'phi_cov': phi_cov_hist
		}