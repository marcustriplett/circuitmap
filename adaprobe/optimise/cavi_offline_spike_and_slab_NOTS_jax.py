import numpy as np

# Jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, while_loop
from jax.ops import index_update
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri

# Experimental loops
from jax.experimental import loops

EPS = 1e-10

@jax.partial(jit, static_argnums=(9, 10, 11))
def cavi_offline_spike_and_slab_NOTS_jax(y, I, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	iters, num_mc_samples, seed):
	"""Offline-mode coordinate ascent variational inference for the adaprobe model.
	"""
	# Initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]

	with loops.Scope() as scope:

		# Declare scope types
		scope.mu 		= jnp.array(mu_prior)
		scope.beta 		= jnp.array(beta_prior)
		scope.alpha 	= jnp.array(alpha_prior)
		scope.shape 	= shape_prior
		scope.rate 		= rate_prior
		scope.phi 		= jnp.array(phi_prior)
		scope.phi_cov 	= jnp.array(phi_cov_prior)
		scope.lam 		= jnp.zeros((N, K))

		# Define history arrays
		scope.mu_hist 		= jnp.zeros((iters, N))
		scope.beta_hist 	= jnp.zeros((iters, N))
		scope.alpha_hist 	= jnp.zeros((iters, N))
		scope.lam_hist 		= jnp.zeros((iters, N, K))
		scope.shape_hist 	= jnp.zeros(iters)
		scope.rate_hist 	= jnp.zeros(iters)
		scope.phi_hist  	= jnp.zeros((iters, N, 2))
		scope.phi_cov_hist 	= jnp.zeros((iters, N, 2, 2))
		
		scope.hist_arrs = [scope.mu_hist, scope.beta_hist, scope.alpha_hist, scope.lam_hist, scope.shape_hist, scope.rate_hist, \
			scope.phi_hist, scope.phi_cov_hist]

		# init key
		scope.key = jax.random.PRNGKey(seed)

		# Iterate CAVI updates
		for it in scope.range(iters):
			scope.beta = update_beta(scope.alpha, scope.lam, scope.shape, scope.rate, beta_prior)
			scope.mu = update_mu(y, scope.mu, scope.beta, scope.alpha, scope.lam, scope.shape, scope.rate, mu_prior, beta_prior, N)
			scope.alpha = update_alpha(y, scope.mu, scope.beta, scope.alpha, scope.lam, scope.shape, scope.rate, alpha_prior, N)
			scope.lam, scope.key = update_lam(y, I, scope.mu, scope.beta, scope.alpha, scope.lam, scope.shape, scope.rate, \
				scope.phi, scope.phi_cov, scope.key, num_mc_samples, N)
			scope.shape, scope.rate = update_sigma(y, scope.mu, scope.beta, scope.alpha, scope.lam, shape_prior, rate_prior)
			(scope.phi, scope.phi_cov), scope.key = update_phi(scope.lam, I, phi_prior, phi_cov_prior, scope.key)

			for hindx, pa in enumerate([scope.mu, scope.beta, scope.alpha, scope.lam, scope.shape, scope.rate, scope.phi, scope.phi_cov]):
				scope.hist_arrs[hindx] = index_update(scope.hist_arrs[hindx], it, pa)

	return scope.mu, scope.beta, scope.alpha, scope.lam, scope.shape, scope.rate, scope.phi, scope.phi_cov, *scope.hist_arrs

# vmap_cavi_offline_spike_and_slab_NOTS_jax = jit(vmap(cavi_offline_spike_and_slab_NOTS_jax, in_axes=(0, 0, *[None]*9)))

@jit
def update_beta(alpha, lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(shape/rate * alpha * jnp.sum(lam, 1) + 1/(beta_prior**2))

@jax.partial(jit, static_argnums=(9))
def update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, N):
	sig = shape/rate
	with loops.Scope() as scope:
		scope.mu = mu
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.all_ids = jnp.arange(N)
		for n in scope.range(N):
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, n - 1), size=N-1)
			scope.mu = index_update(scope.mu, n, (beta[n]**2) * (sig * alpha[n] * jnp.dot(y, lam[n]) - sig * alpha[n] \
				* jnp.dot(lam[n], jnp.sum(jnp.expand_dims(scope.mu[scope.mask] * alpha[scope.mask], 1) * lam[scope.mask], 0)) \
				+ mu_prior[n]/(beta_prior[n]**2)))
	return scope.mu

@jax.partial(jit, static_argnums=(8))
def update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, N):
	with loops.Scope() as scope:
		scope.alpha = alpha
		scope.arg = 0.
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.all_ids = jnp.arange(N)
		for n in scope.range(N):
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, n - 1), size=N-1)
			scope.arg = -2 * mu[n] * jnp.dot(y, lam[n]) + 2 * mu[n] * jnp.dot(lam[n], jnp.sum(jnp.expand_dims(mu[scope.mask] * scope.alpha[scope.mask], 1) \
				* lam[scope.mask], 0)) + (mu[n]**2 + beta[n]**2) * jnp.sum(lam[n])
			scope.alpha = index_update(scope.alpha, n, sigmoid(jnp.log((alpha_prior[n] + EPS)/(1 - alpha_prior[n] + EPS)) - shape/(2 * rate) * scope.arg))
	return scope.alpha

@jax.partial(jit, static_argnums=(11, 12))
def update_lam(y, I, mu, beta, alpha, lam, shape, rate, phi, phi_cov, key, num_mc_samples, N):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	K = I.shape[1]
	with loops.Scope() as scope:
		
		# declare within-scope types
		scope.lam = lam
		scope.all_ids = jnp.arange(N)
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.arg = jnp.zeros(K, dtype=float)
		scope.key, scope.key_next = key, key
		scope.u = jnp.zeros((num_mc_samples, 2))
		scope.mean, scope.sdev = jnp.zeros(2, dtype=float), jnp.zeros(2, dtype=float)
		scope.mc_samps = jnp.zeros((num_mc_samples, 2), dtype=float)
		scope.mcE = jnp.zeros(K)

		for n in scope.range(N):
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, n - 1), size=N-1)
			scope.arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * jnp.sum(jnp.expand_dims(mu[scope.mask] * alpha[scope.mask], 1) * scope.lam[scope.mask], 0) \
			+ (mu[n]**2 + beta[n]**2) * alpha[n]

			# sample truncated normals
			scope.key, scope.key_next = jax.random.split(scope.key)
			scope.u = jax.random.uniform(scope.key, [num_mc_samples, 2])
			scope.mean, scope.sdev = phi[n], jnp.diag(phi_cov[n])
			scope.mc_samps = ndtri(ndtr(-scope.mean/scope.sdev) + scope.u * (1 - ndtr(-scope.mean/scope.sdev))) * scope.sdev + scope.mean

			# monte carlo approximation of expectation
			scope.mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], scope.mc_samps[:, 0], scope.mc_samps[:, 1]), 0)
			# scope.lam = index_update(scope.lam, n, sigmoid(scope.mcE - shape/(2 * rate) * scope.arg * (I[n] > 0))) # require spiking cells to be targeted
			scope.lam = index_update(scope.lam, n, sigmoid(scope.mcE - shape/(2 * rate) * scope.arg))
	return scope.lam, scope.key_next

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
	"""Returns updated sigmoid coefficients estimated using a log-barrier penalty with backtracking Newton's method
	"""
	(posterior, logliks), keys = laplace_approx(lam, phi_prior, phi_cov_prior, I, key) # N keys returned due to vmapped LAs
	return posterior, keys[-1]
	
def _laplace_approx(y, phi_prior, phi_cov, I, key, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	"""Laplace approximation to sigmoid coefficient posteriors $phi$.
	"""

	newton_steps = 10 # need to figure out how to make this dynamic

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
