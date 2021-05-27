import numpy as np
from numba import njit, vectorize, float64, int64
from .utils import sigmoid, get_mask
from scipy.special import ndtr, ndtri
EPS = 1e-15

@njit
def cavi_offline_spike_and_slab_3d_omega(y, stim, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, phi_map_prior, phi_cov_prior, Omega, C, 
	init_t=1e4, t_mult=1e1, t_loops=10, iters=10, verbose=False, newton_steps=10, seed=None, num_mc_samples=5, return_parameter_history=False):
	"""Online-mode coordinate-ascent variational inference for the adaprobe model.

	"""
	L, I = stim
	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prior.shape[0]
	K = len(y)

	# Attenuated power
	I_atten = np.array([[I[k] * np.exp(-(L[k] - C[n]) @ Omega[n] @ (L[k] - C[n])) for k in range(K)] for n in range(N)])

	mu = mu_prior.copy()
	beta = beta_prior.copy()
	alpha = alpha_prior.copy()
	lam = np.zeros((N, K))
	shape = shape_prior
	rate = rate_prior
	phi_map = phi_map_prior.copy()
	phi_cov = phi_cov_prior.copy()

	mask = get_mask(N)
	# lam = update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mask, I_atten, num_mc_samples=num_mc_samples)

	mu_hist 		= np.zeros((iters, N))
	beta_hist 		= np.zeros((iters, N))
	alpha_hist		= np.zeros((iters, N))
	lam_hist 		= np.zeros((iters, N, K))
	shape_hist 		= np.zeros(iters)
	rate_hist 		= np.zeros(iters)
	phi_map_hist 	= np.zeros((iters, N, 2))
	phi_cov_hist 	= np.zeros((iters, N, 2, 2))

	# Iterate CAVI updates
	for it in range(iters):
		beta = update_beta(alpha, lam, shape, rate, beta_prior)
		mu = update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, mask)
		alpha = update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, mask)
		lam = update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mask, I_atten, num_mc_samples=num_mc_samples)
		shape, rate = update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior)
		phi_map, phi_cov = update_phi(lam, phi_map_prior, phi_cov_prior, I_atten, init_t=init_t, t_mult=t_mult, t_loops=t_loops, 
			newton_steps=newton_steps, verbose=verbose)

		mu_hist[it] 		= mu
		beta_hist[it] 		= beta
		alpha_hist[it]		= alpha
		lam_hist[it]		= lam
		shape_hist[it]		= shape
		rate_hist[it] 		= rate
		phi_map_hist[it] 	= phi_map
		phi_cov_hist[it] 	= phi_cov

	return mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, phi_map_hist, phi_cov_hist

@njit
def update_beta(alpha, lam, shape, rate, beta_prior):
	return 1/np.sqrt(shape/rate * alpha * np.sum(lam, 1) + 1/(beta_prior**2))

@njit
def update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, mask):
	N = mu.shape[0]
	sig = shape/rate
	for n in range(N):
		mu[n] = (beta[n]**2) * (sig * alpha[n] * np.dot(y, lam[n]) - sig * alpha[n] \
			* np.dot(lam[n], np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1) * lam[mask[n]], 0)) \
			+ mu_prior[n]/(beta_prior[n]**2))
	return mu

@njit
def update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, mask):
	N = mu.shape[0]
	for n in range(N):
		arg = -2 * mu[n] * np.dot(y, lam[n]) + 2 * mu[n] * np.dot(lam[n], np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1) \
			* lam[mask[n]], 0)) + (mu[n]**2 + beta[n]**2) * np.sum(lam[n])
		alpha[n] = sigmoid(np.log((alpha_prior[n] + EPS)/(1 - alpha_prior[n] + EPS)) - shape/(2 * rate) * arg)
	return alpha

@njit
def update_lam_MAP(y, mu, beta, alpha, lam, shape, rate, phi_map, omega, I, L, C, mask):
	"""Infer latent spike rates using the MAP estimate of the sigmoid coefficients.
	"""
	N, K = lam.shape
	f = np.zeros((N, K))
	for k in range(K):
		f[:, k] = sigmoid(phi_map[:, 0] * I[k] * np.exp(-omega * np.sum(np.square(L[k] - C), 1)) - phi_map[:, 1])
	for n in range(N):
		arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1)) + (mu[n]**2 + beta[n]**2)
		lam[n] = sigmoid(np.log((f[n] + EPS)/(1 - f[n] + EPS)) - shape/(2 * rate) * arg)
	return lam

@njit
def _sample_phi(phi_mapn, phi_covn, num_mc_samples=1):
	"""Returns (num_mc_samples x 2) sample of (non-truncated) phi.
	"""
	samps = np.zeros((num_mc_samples, 2))
	chol = np.linalg.cholesky(phi_covn)
	for n in range(num_mc_samples):
		samps[n] = chol @ np.random.standard_normal(phi_mapn.shape[0]) + phi_mapn
	return samps

@vectorize([float64(float64, float64, float64)], nopython=True)
def _sample_truncated_normal(u, mean, sdev):
	"""Vectorised, JIT-compiled truncated normal samples using scipy's ndtr and ndtri
	"""
	return ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

@njit
def _sample_phi_independent_truncated_normals(phi_mapn, phi_covn, num_mc_samples=5):
	"""Returns (num_mc_samples x 2) sample of phi. Values are sampled from independent univariate truncated normals, due to
	intractability of sampling from truncated multivariate normals.
	"""
	samps = np.zeros((num_mc_samples, 2))
	for i in range(2):
		samps[:, i] = _sample_truncated_normal(np.random.rand(num_mc_samples), phi_mapn[i], np.sqrt(phi_covn[i, i]))
	return samps

@njit
def update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, phi_map, phi_cov, mask, I_atten, num_mc_samples=5):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	N, K = lam.shape
	for n in range(N):
		arg = -2 * y * mu[n] + 2 * mu[n] * np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1) * lam[mask[n]], 0) \
		+ (mu[n]**2 + beta[n]**2)
		mc_samps = _sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=num_mc_samples) # samples of phi for neuron n
		mcE = np.zeros(K) # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			fn = sigmoid(mc_samps[indx, 0] * I_atten[n] - mc_samps[indx, 1])
			fn[fn < EPS] = EPS
			fn[fn > 1 - EPS] = 1 - EPS
			mcE = mcE + np.log((fn)/(1 - fn))
		mcE = mcE/num_mc_samples
		lam[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lam

@njit
def update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior):
	K = len(y)
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (np.sum(np.square(y - np.sum(np.expand_dims(mu * alpha, 1) * lam, 0))) \
		- np.sum(np.square(np.expand_dims(mu * alpha, 1) * lam)) + np.sum(np.expand_dims((mu**2 + beta**2) * alpha, 1) * lam))
	return shape, rate

@njit
def update_phi(lam, phi_prior, phi_cov_prior, I_atten, newton_steps=10, tol=1e-8, init_t=1e2, t_mult=1e1, t_loops=10, verbose=False):
	"""Returns updated sigmoid coefficients estimated using a sequential log-barrier penalty with backtracking Newton's method
	"""
	N = phi_prior.shape[0]
	phi_cov = np.zeros_like(phi_cov_prior)
	phi = phi_prior.copy()
	for n in range(N):
		# Reset t
		t = init_t
		for j in range(t_loops):
			for st in range(newton_steps):
				# Solve barrier method with current t
				phi_new, phi_cov_new = _backtracking_newton_step_with_barrier(phi[n], lam[n], t, phi_prior[n], phi_cov_prior[n], I_atten[n], verbose=verbose)
				if np.mean(np.abs(phi_new - phi[n])) < tol:
					# Newton's method converged
					break
				else:
					phi[n], phi_cov[n] = phi_new, phi_cov_new
			# sharpen log-barrier
			t = t * t_mult
	return phi, phi_cov

@njit
def _backtracking_newton_step_with_barrier(phi, lam, t, phi_prior, phi_cov_prior, I_atten, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	"""Newton's method with backtracking line search. For fixed neuron n.
	"""
	H_inv = np.zeros((2, 2))
	f = sigmoid(phi[0] * I_atten - phi[1])
	phi_cov_prior_inv = np.linalg.inv(phi_cov_prior)

	# grad of negative log-likelihood
	j1 = -np.sum(I_atten * (lam - f))
	j2 = np.sum(lam - f)
	J = np.array([j1, j2]) + phi_cov_prior_inv @ (phi - phi_prior) - 1/(t * phi)
	
	# hessian of negative log-likelihood
	h11 = np.sum(I_atten**2 * f * (1 - f))
	h12 = -np.sum(I_atten * f * (1 - f))
	h21 = h12
	h22 = np.sum(f * (1 - f))
	H = np.array([[h11, h12], [h21, h22]]) + phi_cov_prior_inv + np.diag(1/(t * phi**2))
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	phi_cov_prior_inv_det = np.linalg.det(phi_cov_prior_inv)

	# begin backtracking
	step = 1
	for it in range(max_backtrack_iters):
		lhs = _negloglik_with_barrier(phi + step * v, t, lam, I_atten, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det)
		rhs = _negloglik_with_barrier(phi, t, lam, I_atten, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det) \
		+ backtrack_alpha * step * J.T @ v
		if np.isnan(lhs) or lhs > rhs:
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
def _negloglik_with_barrier(phi, t, lam, I_atten, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det):
	"""Negative log-likelihood, for use with Newton's method. For fixed neuron n.
	"""
	f = sigmoid(phi[0] * I_atten - phi[1])
	f[f < EPS] = EPS
	f[f > 1 - EPS] = 1 - EPS
	nll = -np.sum(lam * np.log(f) + (1 - lam) * np.log(1 - f)) \
	+ 1/2 * (phi - phi_prior) @ phi_cov_prior_inv @ (phi - phi_prior).T + np.log(2 * np.pi) \
	- 1/2 * np.log(phi_cov_prior_inv_det) - np.sum(np.log(phi))/t
	return nll