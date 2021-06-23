import numpy as np
from numba import njit, vectorize, float64, int64
from .utils import sigmoid, get_mask, get_psf_func, get_filt_grid_around_loc
from scipy.special import ndtr, ndtri
EPS = 1e-15

# jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from jax.lax import scan, cond, while_loop 

# @njit
def cavi_offline_spike_and_slab(y, stim, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, eta_prior, eta_cov_prior, cell_grids, 
	init_t=1e4, t_mult=1e1, t_loops=10, iters=10, verbose=False, newton_steps=10, seed=None, lam_update='monte-carlo', num_mc_samples=5, 
	return_parameter_history=False, num_filter_pts_per_dim=4, filter_size=20):
	"""Offline-mode coordinate-ascent variational inference for the adaprobe model.

	"""

	assert lam_update in ['monte-carlo', 'MAP'], "Invalid lambda update specified."

	L, I = stim

	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prior.shape[0]
	K = len(y)

	# Set up grids, filters, and stimuli
	cell_psfs = [get_psf_func(cell_grids[n]) for n in range(N)]
	psfc = np.array([np.c_[cell_psfs[n](I, L), -np.ones(K)] for n in range(N)]) # design matrix
	D = psfc.shape[-1]
	
	mu = mu_prior.copy()
	beta = beta_prior.copy()
	alpha = alpha_prior.copy()
	lam = np.zeros((N, K))
	shape = shape_prior
	rate = rate_prior
	eta = eta_prior.copy()
	eta_cov = eta_cov_prior.copy()

	mask = get_mask(N)
	# lam = update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, eta, eta_cov, mask, psfc, num_mc_samples=num_mc_samples)

	mu_hist 		= np.zeros((iters, N))
	beta_hist 		= np.zeros((iters, N))
	alpha_hist		= np.zeros((iters, N))
	lam_hist 		= np.zeros((iters, N, K))
	shape_hist 		= np.zeros(iters)
	rate_hist 		= np.zeros(iters)
	eta_hist 		= np.zeros((iters, N, D))
	eta_cov_hist 	= np.zeros((iters, N, D, D))

	# Iterate CAVI updates
	for it in range(iters):
		print('iteration %i/%i'%(it+1, iters), end='\r')
		beta = update_beta(alpha, lam, shape, rate, beta_prior)
		mu = update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, mask)
		alpha = update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, mask)
		if lam_update == 'monte-carlo':
			lam = update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, eta, eta_cov, mask, psfc, num_mc_samples=num_mc_samples)
		else:
			lam = update_lam_MAP(y, mu, beta, alpha, lam, shape, rate, eta, eta_cov, mask, psfc)
		shape, rate = update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior)
		eta, eta_cov = update_eta(lam, eta_prior, eta_cov_prior, psfc, newton_steps=newton_steps)

		mu_hist[it] 		= mu
		beta_hist[it] 		= beta
		alpha_hist[it]		= alpha
		lam_hist[it]		= lam
		shape_hist[it]		= shape
		rate_hist[it] 		= rate
		eta_hist[it] 		= eta
		eta_cov_hist[it] 	= eta_cov

	return mu, beta, alpha, lam, shape, rate, eta, eta_cov, mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, eta_hist, eta_cov_hist

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
def _sample_independent_truncated_normals(etan, eta_covn, num_mc_samples=5):
	"""Returns num_mc_samples sample of phi. Values are sampled from independent univariate truncated normals, due to
	intractability of sampling from truncated multivariate normals.
	"""
	edim = etan.shape[0]
	samps = np.zeros((num_mc_samples, edim))
	for i in range(edim):
		samps[:, i] = _sample_truncated_normal(np.random.rand(num_mc_samples), etan[i], np.sqrt(eta_covn[i, i]))
	return samps

@njit
def update_lam_monte_carlo(y, mu, beta, alpha, lam, shape, rate, eta, eta_cov, mask, psfc, num_mc_samples=5):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	N, K = lam.shape
	for n in range(N):
		arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		mc_samps = _sample_independent_truncated_normals(eta[n], eta_cov[n], num_mc_samples=num_mc_samples) # samples of eta for neuron n
		mcE = np.zeros(K) # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			fn = sigmoid(psfc[n] @ mc_samps[indx])
			fn[fn < EPS] = EPS
			fn[fn > 1 - EPS] = 1 - EPS
			mcE = mcE + np.log(fn/(1 - fn) + EPS)
		mcE = mcE/num_mc_samples
		lam[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lam

@njit
def update_lam_MAP(y, mu, beta, alpha, lam, shape, rate, eta, eta_cov, mask, psfc):
	"""Infer latent spike rates using the MAP estimate of the sigmoid coefficients.
	"""
	N, K = lam.shape
	for n in range(N):
		fn = sigmoid(psfc[n] @ eta[n])
		arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * np.sum(np.expand_dims(mu[mask[n]] * alpha[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		lam[n] = sigmoid(np.log((fn + EPS)/(1 - fn + EPS)) - shape/(2 * rate) * arg)
	return lam

@njit
def update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior):
	K = len(y)
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (np.sum(np.square(y - np.sum(np.expand_dims(mu * alpha, 1) * lam, 0))) \
		- np.sum(np.square(np.expand_dims(mu * alpha, 1) * lam)) + np.sum(np.expand_dims((mu**2 + beta**2) * alpha, 1) * lam))
	return shape, rate

def update_eta(y, eta_prior, eta_cov_prior, psfc, newton_steps=15, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	N = y.shape[0]
	posterior, logliks = laplace_approx(y, eta_prior, eta_cov_prior, psfc) # parallel Laplace approximations
	eta, eta_cov = posterior
	return np.array(eta), np.array(eta_cov) # convert to numpy array

""" 
	jax-compiled funcs:
"""

@jit
def _laplace_approx(y, filt_prior, filt_cov, psfc, newton_steps=15, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	"""Laplace approximation to filter posteriors eta.
	"""

	@jit
	def backtrack_cond(carry):
		it, _, lhs, rhs, _, _, _ = carry
		return jnp.logical_and(it < max_backtrack_iters, jnp.logical_or(jnp.isnan(lhs), lhs > rhs))

	@jit
	def backtrack(carry):
		it, step, lhs, rhs, v, J, filt = carry
		it += 1
		step *= backtrack_beta
		lhs, rhs = get_ineq(y, filt, step, v, psfc, t, J, backtrack_alpha)
		return (it, step, lhs, rhs, v, J, filt)
	
	@jit
	def get_ineq(y, filt, step, v, psfc, t, J, backtrack_alpha):
		return negloglik_with_barrier(y, filt + step * v, filt_prior, psfc, prior_prec, t), negloglik_with_barrier(y, filt, filt_prior, psfc, prior_prec, t) + backtrack_alpha * step * J @ v

	@jit
	def get_stepv(filt, t):
		lam = jax.nn.sigmoid(psfc @ filt)
		J = (lam - y) @ psfc - 1/(t * filt) + prior_prec @ (filt - filt_prior) 
		H = jnp.einsum('ij,ik->jk', (lam * (1 - lam))[:, None] * psfc, psfc) + jnp.diag(1/(t*filt**2)) + prior_prec
		H_inv = jnp.linalg.inv(H)
		v = -H_inv @ J
		return v, J, H_inv
	
	@jit
	def newton_step(filt_carry, _):
		filt, _ = filt_carry
		v, J, cov = get_stepv(filt, t)  
		step = 1.
		lhs, rhs = get_ineq(y, filt, step, v, psfc, t, J, backtrack_alpha)
		init_carry = (0, step, lhs, rhs, v, J, filt)
		carry = while_loop(backtrack_cond, backtrack, init_carry)
		_, step, lhs, _, _, _, _ = carry
		filt += step * v
		return (filt, cov), lhs
	
	key = jax.random.PRNGKey(1)
	filt = jax.random.uniform(key, shape=[psfc.shape[1]])
	prior_prec = jnp.linalg.inv(filt_cov)
	newton_steps = 30 # hard-coded for now
	filt_carry = (filt, jnp.zeros((filt.shape[0], filt.shape[0])))
	return scan(newton_step, filt_carry, jnp.arange(newton_steps))

laplace_approx = jit(vmap(_laplace_approx, (0, 0, 0, 0))) # parallel LAs across all cells

@jit
def negloglik_with_barrier(y, filt, filt_prior, psfc, prec, t):
	lam = jax.nn.sigmoid(psfc @ filt)
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) - jnp.sum(jnp.log(filt))/t + 1/2 * (filt - filt_prior) @ prec @ (filt - filt_prior)
