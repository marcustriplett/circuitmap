import numpy as np
from numba import njit
from .utils import sigmoid, get_mask
EPS = 1e-13

@njit
def cavi_offline(y, stim, mu_prior, beta_prior, shape_prior, rate_prior, phi_map_prior, phi_cov_prior, omega, C, 
	iters=10, verbose=False, newton_steps=10, seed=None, lam_update='MAP', num_mc_samples=5, return_parameter_history=False):
	"""Online-mode coordinate-ascent variational inference for the adaprobe model.

	"""
	L, I = stim
	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]
	mu = np.random.normal(0, 1, N)
	beta = np.exp(np.random.normal(0, 1, N))
	lam = np.random.rand(N, K)
	shape = np.random.rand()
	rate = 5 + np.random.rand()
	phi_map = phi_map_prior.copy()
	phi_cov = phi_cov_prior.copy()
	mask = get_mask(N)

	mu_hist 		= np.zeros((iters, N))
	beta_hist 		= np.zeros((iters, N))
	lam_hist 		= np.zeros((iters, N, K))
	shape_hist 		= np.zeros(iters)
	rate_hist 		= np.zeros(iters)
	phi_map_hist 	= np.zeros((iters, N, 2))
	phi_cov_hist 	= np.zeros((iters, N, 2, 2))

	# Iterate CAVI updates
	for it in range(iters):
		beta = update_beta(lam, shape, rate, beta_prior)
		mu = update_mu(y, mu, beta, lam, shape, rate, mu_prior, beta_prior, mask)
		if lam_update == 'MAP':
			lam = update_lam_MAP(y, mu, beta, lam, shape, rate, phi_map, omega, I, L, C, mask)
		elif lam_update == 'monte-carlo':
			lam = update_lam_monte_carlo(y, mu, beta, lam, shape, rate, phi_map, phi_cov, mask, omega, L, I, C, num_mc_samples=num_mc_samples)
		shape, rate = update_sigma(y, mu, beta, lam, shape_prior, rate_prior)
		phi_map, phi_cov = update_phi(lam, phi_map_prior, phi_cov_prior, omega, I, L, C, n_steps=newton_steps, verbose=verbose)

		mu_hist[it] 		= mu
		beta_hist[it] 		= beta
		lam_hist[it]		= lam
		shape_hist[it]		= shape
		rate_hist[it] 		= rate
		phi_map_hist[it] 	= phi_map
		phi_cov_hist[it] 	= phi_cov

	return mu, beta, lam, shape, rate, phi_map, phi_cov, mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, phi_map_hist, phi_cov_hist

@njit
def update_beta(lam, shape, rate, beta_prior):
	return 1/np.sqrt(shape/rate * np.sum(lam, 1) + 1/(beta_prior**2))

@njit
def update_mu(y, mu, beta, lam, shape, rate, mu_prior, beta_prior, mask):
	N = mu.shape[0]
	sig = shape/rate
	for n in range(N):
		mu[n] = (beta[n]**2) * (sig * np.dot(y, lam[n]) - sig * np.dot(lam[n], np.sum(np.expand_dims(mu[mask[n]], 1) * lam[mask[n]], 0)) \
			+ mu_prior[n]/(beta_prior[n]**2))
	return mu

@njit
def update_lam_MAP(y, mu, beta, lam, shape, rate, phi_map, omega, I, L, C, mask):
	N, K = lam.shape
	f = np.zeros((N, K))
	for k in range(K):
		f[:, k] = sigmoid(phi_map[:, 0] * I[k] * np.exp(-omega * np.sum(np.square(L[k] - C), 1)) - phi_map[:, 1])
	for n in range(N):
		arg = -2 * y * mu[n] + 2 * mu[n] * np.sum(np.expand_dims(mu[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		lam[n] = sigmoid(np.log((f[n] + EPS)/(1 - f[n] + EPS)) - shape/(2 * rate) * arg)
	return lam

@njit
def _sample_phi(phi_mapn, phi_covn, num_mc_samples=1):
	"""Returns (num_mc_samples x 2) sample of phi.
	"""
	samps = np.zeros((num_mc_samples, 2))
	chol = np.linalg.cholesky(phi_covn)
	for n in range(num_mc_samples):
		samps[n] = chol @ np.random.standard_normal(phi_mapn.shape[0]) + phi_mapn
	return samps

@njit
def update_lam_monte_carlo(y, mu, beta, lam, shape, rate, phi_map, phi_cov, mask, omega, L, I, C, num_mc_samples=5):
	N, K = lam.shape
	for n in range(N):
		arg = -2 * y * mu[n] + 2 * mu[n] * np.sum(np.expand_dims(mu[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		mc_samps = _sample_phi(phi_map[n], phi_cov[n], num_mc_samples=num_mc_samples) # monte carlo samples of phi for neuron n
		mcE = np.zeros(K) # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			fn = sigmoid(mc_samps[indx, 0] * I * np.exp(-omega[n] * np.sum(np.square(L - C[n]), 1)) - mc_samps[indx, 1])
			mcE = mcE + np.log((fn + EPS)/(1 - fn + EPS))
		mcE = mcE/num_mc_samples
		lam[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lam

@njit
def update_sigma(y, mu, beta, lam, shape_prior, rate_prior):
	K = y.shape[0]
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (np.sum(np.square(y - np.sum(np.expand_dims(mu, 1) * lam, 0))) \
		- np.sum(np.square(np.expand_dims(mu, 1) * lam)) + np.sum(np.expand_dims(mu**2 + beta**2, 1) * lam))
	return shape, rate

@njit
def update_phi(lam, phi_prior, phi_cov_prior, omega, I, L, C, n_steps=10, tol=1e-5, verbose=False):
	N = phi_prior.shape[0]
	phi_cov = np.zeros_like(phi_cov_prior)
	phi = phi_prior.copy()
	for n in range(N):
		for st in range(n_steps):
			phi[n], phi_cov[n] = _backtracking_newton_step_phi(phi[n], lam[n], phi_prior[n], phi_cov_prior[n], omega[n], I, L, C[n], verbose=verbose)
	return phi, phi_cov

@njit
def _backtracking_newton_step_phi(phi, lam, phi_prior, phi_cov_prior, omega, I, L, C, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	"""Newton's method with backtracking line search. For fixed neuron n.
	"""
	m = I * np.exp(-omega * np.sum(np.square(C - L), 1))
	H_inv = np.zeros((2, 2))
	f = sigmoid(phi[0] * m - phi[1])
	phi_cov_prior_inv = np.linalg.inv(phi_cov_prior)

	# grad of negative log-likelihood
	j1 = -np.sum(m * (lam - f))
	j2 = np.sum(lam - f)
	J = np.array([j1, j2]) + phi_cov_prior_inv @ (phi - phi_prior)
	
	# hessian of negative log-likelihood
	h11 = np.sum(m**2 * f * (1 - f))
	h12 = -np.sum(m * f * (1 - f))
	h21 = h12
	h22 = np.sum(f * (1 - f))
	H = np.array([[h11, h12], [h21, h22]]) + phi_cov_prior_inv
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	phi_cov_prior_inv_det = np.linalg.det(phi_cov_prior_inv)

	# begin backtracking
	step = 1
	for it in range(max_backtrack_iters):
		lhs = _negloglik(phi + step * v, lam, m, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det)
		rhs = _negloglik(phi, lam, m, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det) \
		+ backtrack_alpha * step * J.T @ v
		if lhs > rhs:
			# shrink stepsize
			if verbose: print('shrinking step')
			step = backtrack_beta * step
		else:
			# proceed with Newton step
			if verbose: print('step size found', step)
			break
		if verbose and it == max_backtrack_iters - 1:
			print('no step size found')

	return phi + step * v, H_inv

@njit
def _negloglik(phi, lam, m, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det):
	"""Negative log-likelihood, for use with Newton's method. For fixed neuron n.
	"""
	f = sigmoid(phi[0] * m - phi[1])
	nll = -np.sum(lam * np.log(f + EPS) + (1 - lam) * np.log(1 - f + EPS)) \
	+ 1/2 * (phi - phi_prior) @ phi_cov_prior_inv @ (phi - phi_prior).T + np.log(2 * np.pi) \
	- 1/2 * np.log(phi_cov_prior_inv_det)
	return nll