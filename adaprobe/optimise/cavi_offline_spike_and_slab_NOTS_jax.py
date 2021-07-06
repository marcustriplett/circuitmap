import numpy as np

# Jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, while_loop
from jax.ops import index_update
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri

EPS = 1e-10

def cavi_offline_spike_and_slab_NOTS_jax(y, I, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	iters=10, num_mc_samples=10):
	"""Online-mode coordinate ascent variational inference for the adaprobe model.

	"""
	# Initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]

	mu = mu_prior.copy()
	beta = beta_prior.copy()

	alpha = alpha_prior.copy()
	shape = shape_prior
	rate = rate_prior
	phi = phi_prior.copy()
	phi_cov = phi_cov_prior.copy()

	lam = jnp.zeros((N, K))

	# init key
	key = jax.random.PRNGKey(0)

	# Iterate CAVI updates
	for it in range(iters):
		beta = update_beta(alpha, lam, shape, rate, beta_prior)
		mu = update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior)
		alpha = update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior)
		lam, key = update_lam(y, I, mu, beta, alpha, lam, shape, rate, phi, phi_cov, key, num_mc_samples=num_mc_samples)
		shape, rate = update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior)
		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)

	return mu, beta, alpha, lam, shape, rate, phi, phi_cov

@jit
def update_beta(alpha, lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(shape/rate * alpha * jnp.sum(lam, 1) + 1/(beta_prior**2))

@jit # in-place index_updates fast enough?
def update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior):
	N = mu.shape[0]
	sig = shape/rate
	for n in range(N):
		mask = jnp.append(jnp.arange(n), jnp.arange(n + 1, N))
		mu = index_update(mu, n, (beta[n]**2) * (sig * alpha[n] * jnp.dot(y, lam[n]) - sig * alpha[n] \
			* jnp.dot(lam[n], jnp.sum(jnp.expand_dims(mu[mask] * alpha[mask], 1) * lam[mask], 0)) \
			+ mu_prior[n]/(beta_prior[n]**2)))
	return mu

@jit
def update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior):
	N = mu.shape[0]
	for n in range(N):
		mask = jnp.append(jnp.arange(n), jnp.arange(n + 1, N))
		arg = -2 * mu[n] * jnp.dot(y, lam[n]) + 2 * mu[n] * jnp.dot(lam[n], jnp.sum(jnp.expand_dims(mu[mask] * alpha[mask], 1) \
			* lam[mask], 0)) + (mu[n]**2 + beta[n]**2) * jnp.sum(lam[n])
		alpha = index_update(alpha, n, sigmoid(jnp.log((alpha_prior[n] + EPS)/(1 - alpha_prior[n] + EPS)) - shape/(2 * rate) * arg))
	return alpha

def get_trunc_norm_sampler(n_samples):
	def _sampler(key, mean, sdev):
		key, key_next = jax.random.split(key)
		u = jax.random.uniform(key, [n_samples, 1])
		return ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean, key_next
	return jit(_sampler)

@jax.partial(jit, static_argnums=(11,))
def update_lam(y, I, mu, beta, alpha, lam, shape, rate, phi, phi_cov, key, num_mc_samples=10):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	
	# Truncated normal sampler
	sample_phi_independent_truncated_normals = get_trunc_norm_sampler(num_mc_samples) # regenerated every function call

	N = mu.shape[0]
	for n in range(N):
		mask = jnp.append(jnp.arange(n), jnp.arange(n + 1, N))
		arg = -2 * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * jnp.sum(jnp.expand_dims(mu[mask] * alpha[mask], 1) * lam[mask], 0) \
		+ (mu[n]**2 + beta[n]**2) * alpha[n]
		mc_samps, key = sample_phi_independent_truncated_normals(key, phi[n], phi_cov[n]) # samples of phi for neuron n
		num_mc_samples = mc_samps.shape[0]
		mcE = 0 # monte carlo approximation of expectation
		for indx in range(num_mc_samples): ## ### can potentially vectorise this ######
			fn = sigmoid(mc_samps[indx, 0] * I[n] - mc_samps[indx, 1])
			mcE = mcE + jnp.log(fn/(1 - fn))
		mcE = mcE/num_mc_samples
		lam = index_update(lam, n, sigmoid(mcE - shape/(2 * rate) * arg))
	return lam, key

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
	return laplace_approx(lam, phi_prior, phi_cov_prior, I, key)
	
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

	key, key_next = jax.random.PRNGKey(key)
	phi = jnp.array(phi_prior, copy=True)
	prior_prec = jnp.linalg.inv(phi_cov)
	phi_carry = (phi, jnp.zeros((phi.shape[0], phi.shape[0])))
	return scan(newton_step, phi_carry, jnp.arange(newton_steps)), key_next

laplace_approx = jit(vmap(_laplace_approx, (0, 0, 0, 0, None))) # parallel LAs across all cells

@jit
def negloglik_with_barrier(y, phi, phi_prior, prec, I, t):
	lam = sigmoid(phi[0] * I - phi[1])
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) - jnp.sum(jnp.log(phi))/t + 1/2 * (phi - phi_prior) @ prec @ (phi - phi_prior)

