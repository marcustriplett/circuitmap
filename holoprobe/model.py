import numpy as np
from holoprobe import optimise
from holoprobe.optimise.utils import get_filt_grid_around_loc_multi

class Model:
	def __init__(self, cell_locs, num_filter_pts_per_dim=4, filter_dim_len=20, priors=dict()):
		"""Initialise adaprobe model.

		Args:
			cell_locs: 
			priors:

		"""

		self.priors = priors
		self.trial_count = 0
		self.cell_locs = cell_locs
		self.n_presynaptic = cell_locs.shape[0]
		self.cell_grids = get_filt_grid_around_loc_multi(cell_locs, num_points=num_filter_pts_per_dim, dim=filter_dim_len)
		
		# Stimulus/response lists
		self.I = []
		self.L = []
		self.y = []

		# Config priors
		self.priors.setdefault('alpha', 1/2 * np.ones(self.n_presynaptic))
		self.priors.setdefault('shape', 1.)
		self.priors.setdefault('rate', 1.)
		self.priors.setdefault('mu', np.zeros(self.n_presynaptic))
		self.priors.setdefault('beta', 3 * np.ones(self.n_presynaptic))
		eta_prior = np.zeros((self.n_presynaptic, self.cell_grids.shape[1] + 1))
		eta_prior[:, -1] = 6
		self.priors.setdefault('eta', eta_prior)
		eta_cov_prior = np.array([1e-1 * np.diag(np.ones(self.cell_grids.shape[1] + 1)) for _ in range(self.n_presynaptic)])
		eta_cov_prior[:, -1, -1] = 1e-1 # 1e-1 works
		self.priors.setdefault('eta_cov', eta_cov_prior)

		# Set initial state to prior
		self.state = self.priors.copy()

		self.state['lam'] = []

	def update(self, obs, stimuli, method='cavi_online_spike_and_slab', return_params=False, fit_options=dict()):
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

	def fit(self, obs, stimuli, method='cavi_offline_spike_and_slab', fit_options=dict()):
		"""Fit posterior distributions in offline mode.
		"""
		if method == 'cavi':
			self._fit_cavi(obs, stimuli, fit_options)
		elif method == 'cavi_spike_and_slab':
			self._fit_cavi_spike_and_slab(obs, stimuli, fit_options)
		elif method == 'cavi_offline_spike_and_slab':
			self._fit_cavi_offline_spike_and_slab(obs, stimuli, fit_options)
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


	def _fit_cavi_offline_spike_and_slab(self, obs, stimuli, fit_options):
		"""Run CAVI in offline mode with spike-and-slab synapse prior.
		"""
		result = optimise.cavi_offline_spike_and_slab(
			obs, stimuli, self.state['mu'], self.state['beta'], self.state['alpha'], self.state['shape'], 
			self.state['rate'], self.state['eta'], self.state['eta_cov'], self.cell_grids, **fit_options 
		)

		mu, beta, alpha, lam, shape, rate, eta, eta_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		eta_hist, eta_cov_hist = result

		self.state['mu'] 		= mu
		self.state['beta'] 		= beta
		self.state['alpha']		= alpha
		self.state['shape'] 	= shape
		self.state['rate'] 		= rate
		self.state['eta']	 	= eta
		self.state['eta_cov'] 	= eta_cov
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
			'eta': eta_hist,
			'eta_cov': eta_cov_hist
		}

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