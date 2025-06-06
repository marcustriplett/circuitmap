import numpy as np
from functools import partial

# Jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan, while_loop, fori_loop
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri
from jax import config; config.update("jax_enable_x64", True)
from tqdm import trange
# from .pava import _isotonic_regression, simultaneous_isotonic_regression

EPS = 1e-10

def cavi_sns(y_psc, I, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, iters=50, 
	num_mc_samples=100, seed=1, y_xcorr_thresh=1e-2, minimum_spike_count=3, save_histories=True):
	''' Coordinate-ascent variational inference for spike-and-slab model.
	'''
	print('Running coordinate-ascent variational inference for spike-and-slab model.')
	y = np.trapz(y_psc, axis=-1)
	K = y.shape[0]
	# lam_mask = jnp.array([jnp.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh
	lam_mask = jnp.ones(K)

	lam = np.zeros_like(I)
	lam[I > 0] = 0.95
	lam = lam * lam_mask
	spont_rate = 0.

	# Initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]
	powers = jnp.array(np.unique(I)[1:])

	# Declare scope types
	mu 			= jnp.array(mu_prior)
	beta 		= jnp.array(beta_prior)
	alpha 		= jnp.array(alpha_prior)
	shape 		= jnp.array(shape_prior)
	rate 		= jnp.array(rate_prior)
	phi 		= jnp.array(phi_prior)
	phi_cov 	= jnp.array(phi_cov_prior)

	# Define history arrays
	if save_histories:
		cpu = jax.devices('cpu')[0]

		mu_hist 		= np.zeros((iters, N))
		beta_hist 		= np.zeros((iters, N))
		alpha_hist 		= np.zeros((iters, N))
		lam_hist 		= np.zeros((iters, N, K))
		shape_hist 		= np.zeros((iters, K))
		rate_hist 		= np.zeros((iters, K))
		phi_hist  		= np.zeros((iters, N, 2))
		phi_cov_hist 	= np.zeros((iters, N, 2, 2))
		
		hist_arrs = [mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
			phi_hist, phi_cov_hist]

		# move hist arrays to CPU
		hist_arrs = [jax.device_put(ha, cpu) for ha in hist_arrs]

	else:
		hist_arrs = [None] * 8
		
	# init key
	key = jax.random.PRNGKey(seed)

	# Iterate CAVI updates
	for it in trange(iters):
		beta 				= update_beta(alpha, lam, shape, rate, beta_prior)
		mu, key 			= update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, N, key)
		alpha, key 			= update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, N, key)
		lam, key 			= update_lam(y, I, mu, beta, alpha, lam, shape, rate, \
								phi, phi_cov, lam_mask, key, num_mc_samples, N, minimum_spike_count)
		shape, rate 		= update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior)
		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)

		if save_histories:
			for hindx, pa in enumerate([mu, beta, alpha, lam, shape, rate, phi, phi_cov]):
				hist_arrs[hindx] = hist_arrs[hindx].at[it].set(pa)

	return (mu, beta, alpha, lam, shape, rate, phi, phi_cov, *hist_arrs)

@jit
def update_beta(alpha, lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(alpha * jnp.sum(shape/rate * lam, 1) + 1/(beta_prior**2))

@partial(jit, static_argnums=(9))
def update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, N, key):
	''' Update based on solving E_q(Z-mu_n)[ln p(y, Z)]
	'''
	sig = shape/rate
	update_order = jax.random.choice(key, N, [N], replace=False) # randomize update order
	all_ids = jnp.arange(N)

	def body_fun(m, mu_vector):
		n = update_order[m]
		mask = jnp.unique(jnp.where(all_ids != n, all_ids, jnp.mod(n - 1, N)), size=N-1)
		mu_n = (beta[n]**2) * (alpha[n] * jnp.dot(sig * y, lam[n]) - alpha[n] \
			* jnp.dot(sig * lam[n], jnp.sum(jnp.expand_dims(mu_vector[mask] * alpha[mask], 1) * lam[mask], 0)) \
			+ mu_prior[n]/(beta_prior[n]**2))

		return mu_vector.at[n].set(mu_n)

	mu = fori_loop(0, N, body_fun, mu)
	key, _ = jax.random.split(key)

	return mu, key

@partial(jit, static_argnums=(8))
def update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, N, key):
	update_order = jax.random.choice(key, N, [N], replace=False) # randomize update order
	all_ids = jnp.arange(N)

	def body_fun(m, alpha_vector):
		n = update_order[m]
		mask = jnp.unique(jnp.where(all_ids != n, all_ids, jnp.mod(n - 1, N)), size=N-1) 
		arg = -2 * mu[n] * jnp.dot(y, lam[n]) + 2 * mu[n] * jnp.dot(lam[n], jnp.sum(jnp.expand_dims(mu[mask] * alpha_vector[mask], 1) \
			* lam[mask], 0)) + (mu[n]**2 + beta[n]**2) * jnp.sum(lam[n])
		return alpha_vector.at[n].set(sigmoid(jnp.log((alpha_prior[n] + EPS)/(1 - alpha_prior[n] + EPS)) - shape/(2 * rate) * arg))

	alpha = fori_loop(0, N, body_fun, alpha)
	key, _ = jax.random.split(key)
	return alpha, key

@partial(jit, static_argnums=(12, 13)) # lam_mask[k] = 1 if xcorr(y_psc[k]) > thresh else 0.
def update_lam(y, I, mu, beta, alpha, lam, shape, rate, phi, phi_cov, lam_mask, key, num_mc_samples, N, minimum_spike_count):
	''' Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	'''
	K = I.shape[1]
	update_order = jax.random.choice(key, N, [N], replace=False) # randomize update order
	all_ids = jnp.arange(N)

	def body_fun(m, carry):
		lam_vector, current_key = carry
		n = update_order[m]

		mask = jnp.unique(jnp.where(all_ids != n, all_ids, jnp.mod(n - 1, N)), size=N-1)
		arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * jnp.sum(jnp.expand_dims(mu[mask] * alpha[mask], 1) * lam_vector[mask], 0) \
		+ (mu[n]**2 + beta[n]**2) * alpha[n]

		# sample truncated normals
		key, key_next = jax.random.split(key)
		u = jax.random.uniform(key, [num_mc_samples, 2])
		mean, sdev = phi[n], jnp.diag(phi_cov[n])
		mc_samps = ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

		# monte carlo approximation of expectation
		mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], mc_samps[:, 0], mc_samps[:, 1]), 0)
		est_lam = lam_mask * (I[n] > 0) * sigmoid(mcE - shape/(2 * rate) * arg) # require spiking cells to be targeted
		new_lam_vector = lam_vector.at[n].set(est_lam * (jnp.sum(est_lam) >= minimum_spike_count))

		return (new_lam_vector, key_next)

	(lam, key) = fori_loop(0, N, body_fun, (lam, key))
	return lam, key

def _eval_lam_update_monte_carlo(I, phi_0, phi_1):
	fn = sigmoid(phi_0 * I - phi_1)
	return jnp.log(fn/(1 - fn))
_vmap_eval_lam_update_monte_carlo = jit(vmap(_eval_lam_update_monte_carlo, in_axes=(None, 0, 0)))

@jit
def update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior):
	K = y.shape[0]
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (jnp.sum(jnp.square(y - np.sum(jnp.expand_dims(mu * alpha, 1) * lam, 0))) \
		- jnp.sum(jnp.square(jnp.expand_dims(mu * alpha, 1) * lam)) + jnp.sum(jnp.expand_dims((mu**2 + beta**2) * alpha, 1) * lam))
	return shape, rate

@jit
def update_phi(lam, I, phi_prior, phi_cov_prior, key):
	''' Returns updated sigmoid coefficients estimated using a log-barrier penalty with backtracking Newton's method
	'''
	(posterior, logliks), keys = laplace_approx(lam, phi_prior, phi_cov_prior, I, key) # N keys returned due to vmapped LAs
	return posterior, keys[-1]
	
def _laplace_approx(y, phi_prior, phi_cov, I, key, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	''' Laplace approximation to sigmoid coefficient posteriors phi.
	'''

	newton_steps = 10 # could make this dynamic

	def backtrack_cond(carry):
		it, _, lhs, rhs, _, _, _ = carry
		return jnp.logical_and(it < max_backtrack_iters, jnp.logical_or(jnp.isnan(lhs), lhs > rhs))

	def backtrack(carry):
		it, step, lhs, rhs, v, J, phi = carry
		it += 1
		step *= backtrack_beta
		lhs, rhs = get_ineq(y, phi, step, v, t, J, backtrack_alpha)
		return (it, step, lhs, rhs, v, J, phi)

	def get_ineq(y, phi, step, v, t, J, backtrack_alpha):
		return negloglik_with_barrier(y, phi + step * v, phi_prior, prior_prec, I, t), \
			negloglik_with_barrier(y, phi, phi_prior, prior_prec, I, t) + backtrack_alpha * step * J @ v

	def get_stepv(phi, t):
		f = sigmoid(phi[0] * I - phi[1])

		# grad of negative log-likelihood
		j1 = -jnp.sum(I * (y - f))
		j2 = jnp.sum(y - f)
		J = jnp.array([j1, j2]) + prior_prec @ (phi - phi_prior) - 1/(t * phi)

		# hessian of negative log-likelihood
		h11 = jnp.sum(I**2 * f * (1 - f))
		h12 = -jnp.sum(I * f * (1 - f))
		h21 = h12
		h22 = jnp.sum(f * (1 - f))
		H = jnp.array([[h11, h12], [h21, h22]]) + prior_prec + jnp.diag(1/(t * phi**2))

		H_inv = jnp.linalg.inv(H)
		v = -H_inv @ J
		return v, J, H_inv

	def newton_step(phi_carry, _):
		phi, _ = phi_carry
		v, J, cov = get_stepv(phi, t)  
		step = 1.
		lhs, rhs = get_ineq(y, phi, step, v, t, J, backtrack_alpha)
		init_carry = (0, step, lhs, rhs, v, J, phi)
		carry = while_loop(backtrack_cond, backtrack, init_carry)
		_, step, lhs, _, _, _, _ = carry
		phi += step * v
		return (phi, cov), lhs

	key, key_next = jax.random.split(key)
	phi = jnp.array(phi_prior, copy=True)
	prior_prec = jnp.linalg.inv(phi_cov)
	phi_carry = (phi, jnp.zeros((phi.shape[0], phi.shape[0])))
	return scan(newton_step, phi_carry, jnp.arange(newton_steps)), key_next

laplace_approx = jit(vmap(_laplace_approx, (0, 0, 0, 0, None))) # parallel LAs across all cells

@jit
def negloglik_with_barrier(y, phi, phi_prior, prec, I, t):
	lam = sigmoid(phi[0] * I - phi[1])
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) - jnp.sum(jnp.log(phi))/t + 1/2 * (phi - phi_prior) @ prec @ (phi - phi_prior)
