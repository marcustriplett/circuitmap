import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.stats import linregress

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

# Jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.lax import scan, while_loop
from jax.ops import index_update
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri

from jax.config import config; config.update("jax_enable_x64", True)

# Experimental loops
from jax.experimental import loops

EPS = 1e-10

def mbcs_spike_weighted_var_with_sigmoid_rf(obs, I, mu_prior, beta_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, iters=50,
	num_mc_samples=50, seed=0, y_xcorr_thresh=0.05, penalty=5e0, lam_masking=False, scale_factor=0.5,
	max_penalty_iters=10, max_lasso_iters=100, warm_start_lasso=True, constrain_weights='positive',
	verbose=False, learn_noise=False, init_lam=None, learn_lam=True, phi_delay=-1, phi_thresh=0.09,
	minimum_spike_count=1, noise_scale=0.5, num_mc_samples_noise_model=10, minimum_maximal_spike_prob=0.2):
	"""Offline-mode coordinate ascent variational inference for the adaprobe model.
	"""
	if lam_masking:
		y, y_psc = obs
		K = y.shape[0]

		# Setup lam mask
		lam_mask = (np.array([np.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh)

	else:
		y = obs
		K = y.shape[0]
		lam_mask = np.ones(K)

	# Initialise new params
	N = mu_prior.shape[0]

	# Declare scope types
	mu 			= jnp.array(mu_prior)
	beta 		= jnp.array(beta_prior)
	shape 		= shape_prior
	rate 		= rate_prior
	phi 		= jnp.array(phi_prior)
	phi_cov 	= jnp.array(phi_cov_prior)
	
	# Spike initialisation
	if init_lam is None:
		lam = np.zeros_like(I) 
		if lam_masking:
			lam[I > 0] = 0.95
			lam = lam * lam_mask
		else:
			lam[I > 0] = 0.5
	else:
		lam = init_lam

	lam = jnp.array(lam)

	# Define history arrays
	mu_hist 		= jnp.zeros((iters, N))
	beta_hist 		= jnp.zeros((iters, N))
	lam_hist 		= jnp.zeros((iters, N, K))
	shape_hist 		= jnp.zeros((iters, K))
	rate_hist 		= jnp.zeros((iters, K))
	phi_hist  		= jnp.zeros((iters, N, 2))
	phi_cov_hist 	= jnp.zeros((iters, N, 2, 2))

	hist_arrs = [mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist]

	# init key
	key = jax.random.PRNGKey(seed)

	# init spike prior
	# spike_prior = np.zeros_like(lam)
	# spike_prior[lam > 0] = 0.5

	# Iterate CAVI updates
	for it in tqdm(range(iters), desc='CAVI', leave=False):
		beta = update_beta(lam, shape, rate, beta_prior)
		mu = update_mu_constr_l1(y, mu, lam, shape, rate, penalty=penalty, scale_factor=scale_factor, 
			max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, warm_start_lasso=warm_start_lasso, 
			constrain_weights=constrain_weights, verbose=verbose)
		update_order = np.random.choice(N, N, replace=False)
		lam, key = update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, update_order, key, num_mc_samples, N)
		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)

		# lam = update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N)
		
		receptive_field, spike_prior = update_isotonic_receptive_field(lam, I)
		mu, lam = isotonic_filtering(mu, lam, I, receptive_field, minimum_spike_count=minimum_spike_count, minimum_maximal_spike_prob=minimum_maximal_spike_prob)
		shape, rate = update_noise(y, mu, beta, lam, noise_scale=noise_scale, num_mc_samples=num_mc_samples_noise_model)

		# record history
		for hindx, pa in enumerate([mu, beta, lam, shape, rate, phi, phi_cov]):
			hist_arrs[hindx] = index_update(hist_arrs[hindx], it, pa)

	return mu, beta, lam, shape, rate, phi, phi_cov, *hist_arrs

def update_noise(y, mu, beta, lam, noise_scale=0.5, num_mc_samples=10, eps=1e-4):
	N, K = lam.shape
	std = beta * (mu != 0)
	w_samps = np.random.normal(mu, std, [num_mc_samples, N])
	s_samps = (np.random.rand(num_mc_samples, N, K) <= lam[None, :, :]).astype(float)
	mc_ws_sq = np.mean([(w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0)
	mc_recon_err = np.mean([(y - w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0)
	shape = noise_scale**2 * mc_ws_sq + 1/2
	rate = noise_scale * mu @ lam + 1/2 * mc_recon_err + eps
	return shape, rate

def isotonic_filtering(mu, lam, I, isotonic_receptive_field, minimum_spike_count=1, minimum_maximal_spike_prob=0.2):
	# Enforce minimum maximal spike probability
	disc_locs = np.where(isotonic_receptive_field[:, -1] < minimum_maximal_spike_prob)[0]
	mu = index_update(mu, disc_locs, 0.)
	lam = index_update(lam, disc_locs, 0.)

	# Filter connection vector via spike counts
	spks = np.array([len(np.where(lam[n] >= 0.5)[0]) for n in range(mu.shape[0])])
	few_spk_locs = np.where(spks < minimum_spike_count)[0]
	mu = index_update(mu, few_spk_locs, 0.)
	lam = index_update(lam, few_spk_locs, 0.)

	return mu, lam

def update_isotonic_receptive_field(lam, I):
	N, K = lam.shape

	powers = np.unique(I) # includes zero
	n_powers = len(powers)
	inferred_spk_probs = np.zeros((N, n_powers))
	receptive_field = np.zeros((N, n_powers))
	isotonic_regressor = IsotonicRegression(y_min=0, y_max=1, increasing=True)
	spike_prior = np.zeros((N, K))

	for n in range(N):
		for p, power in enumerate(powers[1:]):
			locs = np.where(I[n] == power)[0]
			if locs.shape[0] > 0:
				inferred_spk_probs[n, p + 1] = np.mean(lam[n, locs])

		isotonic_regressor.fit(powers, inferred_spk_probs[n])
		receptive_field[n] = isotonic_regressor.f_(powers)
		spike_prior[n] = isotonic_regressor.f_(I[n])
	return receptive_field, spike_prior

@jit
def update_beta(lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(jnp.sum((shape/rate)[None, :] * lam, 1) + 1/(beta_prior**2))

@jax.partial(jit, static_argnums=(8))
def update_mu(y, mu, beta, lam, shape, rate, mu_prior, beta_prior, N):
	"""Update based on solving E_q(Z-mu_n)[ln p(y, Z)]"""

	sig = shape/rate
	with loops.Scope() as scope:
		scope.mu = mu
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.all_ids = jnp.arange(N)
		for n in scope.range(N):
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, n - 1), size=N-1)
			scope.mu = index_update(scope.mu, n, (beta[n]**2) * (sig * jnp.dot(y, lam[n]) - sig \
				* jnp.dot(lam[n], jnp.sum(jnp.expand_dims(scope.mu[scope.mask], 1) * lam[scope.mask], 0)) \
				+ mu_prior[n]/(beta_prior[n]**2)))
	return scope.mu

def update_mu_constr_l1(y, mu, Lam, shape, rate, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, \
	warm_start_lasso=False, constrain_weights='positive', verbose=False, tol=1e-5):
	""" Constrained L1 solver with iterative penalty shrinking
	"""
	if verbose:
		print(' ====== Updating mu via constrained L1 solver with iterative penalty shrinking ======')
	N, K = Lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	LamT = Lam.T
	positive = constrain_weights in ['positive', 'negative']
	lasso = Lasso(alpha=penalty, fit_intercept=False, max_iter=max_lasso_iters, warm_start=warm_start_lasso, positive=positive)
	
	if constrain_weights == 'negative':
		# make sensing matrix and weight warm-start negative
		LamT = -LamT
		mu = -mu
	lasso.coef_ = np.array(mu)

	err_prev = 0
	for it in range(max_penalty_iters):
		# iteratively shrink penalty until constraint is met
		if verbose:
			print('penalty iter: ', it)
			print('current penalty: ', lasso.alpha)

		lasso.fit(LamT, y)
		coef = lasso.coef_
		err = np.sqrt(np.sum(np.square(y - LamT @ coef)))

		if verbose:
			print('lasso err: ', err)
			print('constr: ', constr)
			print('')

		if err <= constr:
			if verbose:
				print(' ==== converged on iteration: %i ===='%it)
			break
		elif np.abs(err - err_prev) < tol:
			if verbose:
				print(' !!! converged without meeting constraint on iteration: %i !!!'%it)
			break
		else:
			penalty *= scale_factor # exponential backoff
			lasso.alpha = penalty
			if it == 0:
				lasso.warm_start = True

	if constrain_weights == 'negative':
		return -coef
	else:
		return coef

def update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""

	K = I.shape[1]
	all_ids = jnp.arange(N)

	for m in range(N):
		n = update_order[m]
		mask = jnp.unique(jnp.where(all_ids != n, all_ids, n - 1), size=N-1)
		arg = -2 * y * mu[n] + 2 * mu[n] * jnp.sum(jnp.expand_dims(mu[mask], 1) * lam[mask], 0) \
		+ (mu[n]**2 + beta[n]**2)
		lam = index_update(lam, n, lam_mask * (I[n] > 0) * sigmoid(spike_prior[n] - shape/(2 * rate) * arg)) # require spiking cells to be targeted

	return lam

@jax.partial(jit, static_argnums=(12, 13)) # lam_mask[k] = 1 if xcorr(y_psc[k]) > thresh else 0.
def update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, update_order, key, num_mc_samples, N):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	K = I.shape[1]
	with loops.Scope() as scope:
		
		# declare within-scope types
		scope.lam = lam
		scope.all_ids = jnp.arange(N)
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.arg = jnp.zeros(K, dtype=float)
		scope.key, scope.key_next = key, key # scope.key_next needs to be initiated outside of loop
		scope.u = jnp.zeros((num_mc_samples, 2))
		scope.mean, scope.sdev = jnp.zeros(2, dtype=float), jnp.zeros(2, dtype=float)
		scope.mc_samps = jnp.zeros((num_mc_samples, 2), dtype=float)
		scope.mcE = jnp.zeros(K)

		for m in scope.range(N):
			n = update_order[m]
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, n - 1), size=N-1)
			scope.arg = -2 * y * mu[n] + 2 * mu[n] * jnp.sum(jnp.expand_dims(mu[scope.mask], 1) * scope.lam[scope.mask], 0) \
			+ (mu[n]**2 + beta[n]**2)

			# sample truncated normals
			scope.key, scope.key_next = jax.random.split(scope.key)
			scope.u = jax.random.uniform(scope.key, [num_mc_samples, 2])
			scope.mean, scope.sdev = phi[n], jnp.diag(phi_cov[n])
			scope.mc_samps = ndtri(ndtr(-scope.mean/scope.sdev) + scope.u * (1 - ndtr(-scope.mean/scope.sdev))) * scope.sdev + scope.mean

			# monte carlo approximation of expectation
			scope.mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], scope.mc_samps[:, 0], scope.mc_samps[:, 1]), 0)
			
			scope.lam = index_update(scope.lam, n, lam_mask * (I[n] > 0) * sigmoid(scope.mcE - shape/(2 * rate) * scope.arg)) # require spiking cells to be targeted
			
			# scope.lam = index_update(scope.lam, n, lam_mask * (I[n] > 0) * sigmoid(scope.mcE - shape/(2 * rate) * scope.arg \
			# 	+ coact_penalty * jnp.sum(scope.lam >= 0.5, 0)))

			# scope.lam = index_update(scope.lam, n, scope.lam[n] * lam_mask)
	return scope.lam, scope.key_next

def _eval_lam_update_monte_carlo(I, phi_0, phi_1):
	fn = sigmoid(phi_0 * I - phi_1)
	return jnp.log(fn/(1 - fn))
_vmap_eval_lam_update_monte_carlo = jit(vmap(_eval_lam_update_monte_carlo, in_axes=(None, 0, 0)))

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
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) - jnp.sum(jnp.log(phi))/t \
		+ 1/2 * (phi - phi_prior) @ prec @ (phi - phi_prior)
