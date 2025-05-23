import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.stats import linregress
from functools import partial

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
from jax.lax import scan, while_loop, fori_loop
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri

from jax import config; config.update("jax_enable_x64", True)

EPS = 1e-10

def mbcs(y_psc, I, mu_prior, beta_prior, shape_prior, rate_prior, iters=50, 
	num_mc_samples=100, seed=0, y_xcorr_thresh=0.05, penalty=5e0, scale_factor=0.5, max_penalty_iters=10, 
	max_lasso_iters=100, warm_start_lasso=True, constrain_weights='positive', verbose=False, 
	learn_noise=False, init_lam=None, learn_lam=True, delay_spont_estimation=1, minimum_spike_count=1, noise_scale=0.5, 
	num_mc_samples_noise_model=10, minimum_maximal_spike_prob=0.2, orthogonal_outliers=True, outlier_penalty=5e1, 
	init_spike_prior=0.75, outlier_tol=0.05, spont_rate=0, lam_mask_fraction=0.05):
	"""Offline-mode coordinate ascent variational inference for the circuitmap model.
	"""
	print('Running model-based compressed sensing algorithm with isotonic regularisation and spike-dependent noise.')

	y = np.trapz(y_psc, axis=-1)
	K = y.shape[0]

	# Setup lam mask
	lam_mask = (np.array([np.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh)
	lam_mask[np.max(y_psc, axis=1) < lam_mask_fraction * np.max(y_psc)] = 0 # mask events that are too small

	# Initialise new params
	N = mu_prior.shape[0]
	mu = np.random.lognormal(1, 1, N)

	# Declare scope types
	mu 					= jnp.array(mu)
	beta 				= jnp.array(beta_prior)
	shape 				= shape_prior
	rate 				= rate_prior
	z 					= np.zeros(K)
	receptive_fields 	= None
	
	# Spike initialisation
	lam 		= np.zeros_like(I) 
	lam[I > 0] 	= init_spike_prior
	lam 		= lam * lam_mask
	lam 		= jnp.array(lam) # move to DeviceArray
	tar_matrix 	= (I != 0.)

	# Define history arrays
	# % Will need to be disabled for very large matrices
	mu_hist 		= jnp.zeros((iters, N))
	beta_hist 		= jnp.zeros((iters, N))
	lam_hist 		= jnp.zeros((iters, N, K))
	shape_hist 		= jnp.zeros((iters, K))
	rate_hist 		= jnp.zeros((iters, K))
	z_hist 			= jnp.zeros((iters, K))

	hist_arrs = [mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, z_hist]

	# init key
	key = jax.random.PRNGKey(seed)

	# init spike prior
	spike_prior = lam.copy()

	# Iterate CAVI updates
	for it in tqdm(range(iters), leave=True):
		beta = update_beta(lam, shape, rate, beta_prior)
		# ignore z during mu and lam updates
		mu, lam = update_mu_constr_l1(y, mu, lam, shape, rate, penalty=penalty, scale_factor=scale_factor, 
			max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, warm_start_lasso=warm_start_lasso, 
			constrain_weights=constrain_weights, verbose=verbose)

		update_order = np.random.choice(N, N, replace=False)
		lam = update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N)
		receptive_fields, spike_prior = update_isotonic_receptive_field(lam, I)
		mu, lam = isotonic_filtering(mu, lam, I, receptive_fields, minimum_spike_count=minimum_spike_count, minimum_maximal_spike_prob=minimum_maximal_spike_prob + spont_rate)
		shape, rate = update_noise(y, mu, beta, lam, noise_scale=noise_scale, num_mc_samples=num_mc_samples_noise_model)

		if it > delay_spont_estimation:
			z = update_z_l1_with_residual_tolerance(y, mu, lam, lam_mask, penalty=outlier_penalty, scale_factor=scale_factor,
				max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, verbose=verbose, 
				orthogonal=orthogonal_outliers, tol=outlier_tol)
			spont_rate = np.mean(z != 0)

		# record history
		for hindx, pa in enumerate([mu, beta, lam, shape, rate, z]):
			# hist_arrs[hindx] = index_update(hist_arrs[hindx], it, pa)
			hist_arrs[indx] = hist_arrs[hindx].at[it].set(pa)

	return (mu, beta, lam, shape, rate, z, receptive_fields, *hist_arrs)

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
	mu = mu.at[disc_locs].set(0.)
	lam = lam.at[disc_locs].set(0.)

	# Filter connection vector via spike counts
	spks = np.array([len(np.where(lam[n] >= 0.5)[0]) for n in range(mu.shape[0])])
	few_spk_locs = np.where(spks < minimum_spike_count)[0]
	mu = mu.at[few_spk_locs].set(0.)
	lam = lam.at[few_spk_locs].set(0.)

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

def update_mu_constr_l1(y, mu, lam, shape, rate, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, \
	warm_start_lasso=False, constrain_weights='positive', verbose=False, tol=1e-5):
	''' Constrained L1 solver with iterative penalty shrinking
	'''
	if verbose:
		print(' ====== Updating mu via constrained L1 solver with iterative penalty shrinking ======')
	N, K = lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	mu, lam = np.array(mu), np.array(lam) # cast to ndarray

	lamT = lam.T
	positive = constrain_weights in ['positive', 'negative']
	lasso = Lasso(alpha=penalty, fit_intercept=False, max_iter=max_lasso_iters, warm_start=warm_start_lasso, positive=positive)
	
	if constrain_weights == 'negative':
		# make sensing matrix and weight warm-start negative
		lamT = -lamT
		mu = -mu
	lasso.coef_ = mu

	err_prev = 0
	for it in range(max_penalty_iters):
		# iteratively shrink penalty until constraint is met
		if verbose:
			print('penalty iter: ', it)
			print('current penalty: ', lasso.alpha)

		lasso.fit(lamT, y)
		coef = lasso.coef_
		err = np.sqrt(np.sum(np.square(y - lamT @ coef)))

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

	# zero-out spikes for disconnected cells
	lam[np.where(mu) == 0] = 0

	if constrain_weights == 'negative':
		return -coef, lam
	else:
		return coef, lam

def update_z_l1_with_residual_tolerance(y, mu, lam, lam_mask, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, 
	verbose=False, orthogonal=True, tol=0.05):
	''' Soft thresholding with iterative penalty shrinkage
	'''
	if verbose:
		print(' ==== Updating z via soft thresholding with iterative penalty shrinking ==== ')

	N, K = lam.shape
	resid = np.array(y - lam.T @ mu)

	for it in range(max_penalty_iters):
		# iteratively shrink penalty until constraint is met
		if verbose:
			print('penalty iter: ', it)
			print('current penalty: ', penalty)
		
		z = np.zeros(K)
		hard_thresh_locs = np.where(resid < penalty)[0]
		soft_thresh_locs = np.where(resid >= penalty)[0]
		z[hard_thresh_locs] = 0
		z[soft_thresh_locs] = resid[soft_thresh_locs] - penalty
		z[z < 0] = 0

		if orthogonal:
			# enforce orthogonality
			z[np.any(lam >= 0.5, axis=0)] = 0

		z = z * lam_mask
		err = np.sum(np.square(resid - z))/np.sum(np.square(y))

		if verbose:
			print('soft thresh err: ', err)
			print('tol: ', tol)
			print('')

		if err <= tol:
			if verbose:
				print(' ==== converged on iteration: %i ==== '%it)
			break
		else:
			penalty *= scale_factor # exponential backoff
		
	return z

def update_z_constr_l1(y, mu, lam, shape, rate, lam_mask, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, 
	verbose=False, orthogonal=True):
	''' Soft thresholding with iterative penalty shrinkage
	'''
	if verbose:
		print(' ==== Updating z via soft thresholding with iterative penalty shrinking ==== ')

	N, K = lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	resid = np.array(y - lam.T @ mu) # copy to np array, possible memory overhead problem here

	for it in range(max_penalty_iters):
		# iteratively shrink penalty until constraint is met
		if verbose:
			print('penalty iter: ', it)
			print('current penalty: ', penalty)
		
		z = np.zeros(K)
		hard_thresh_locs = np.where(resid < penalty)[0]
		soft_thresh_locs = np.where(resid >= penalty)[0]
		z[hard_thresh_locs] = 0
		z[soft_thresh_locs] = resid[soft_thresh_locs] - penalty
		z[z < 0] = 0

		if orthogonal:
			# enforce orthogonality
			z[np.any(lam >= 0.5, axis=0)] = 0

		z = z * lam_mask
		err = np.sqrt(np.sum(np.square(resid - z)))

		if verbose:
			print('soft thresh err: ', err)
			print('constr: ', constr)
			print('')

		if err <= constr:
			if verbose:
				print(' ==== converged on iteration: %i ==== '%it)
			break
		else:
			penalty *= scale_factor # exponential backoff
		
	return z

def update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N):
	''' Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	'''

	K = I.shape[1]
	all_ids = jnp.arange(N)

	for m in range(N):
		n = update_order[m]
		if mu[n] != 0:
			# mask = jnp.unique(jnp.where(all_ids != n, all_ids, n - 1), size=N-1)
			mask = jnp.array(np.delete(all_ids, n)).squeeze()
			arg = -2 * y * mu[n] + 2 * mu[n] * jnp.sum(jnp.expand_dims(mu[mask], 1) * lam[mask], 0) \
			+ (mu[n]**2 + beta[n]**2)
			lam = lam.at[n].set(lam_mask * (I[n] > 0) * (mu[n] != 0) * sigmoid(spike_prior[n] - shape/(2 * rate) * arg))

	return lam

@partial(jit, static_argnums=(12, 13))
def update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, update_order, key, num_mc_samples, N):
	''' JIT-compiled inference of latent spikes using Monte Carlo samples of sigmoid coefficients.
	'''
	K = I.shape[1]
	all_ids = jnp.arange(N)

	def body_fun(m, state):
		lam_vector, key = state
		n = update_order[m]
		mask = jnp.unique(jnp.where(all_ids != n, all_ids, n - 1), size=N-1)
		arg = -2 * y * mu[n] + 2 * mu[n] * jnp.sum(jnp.expand_dims(mu[mask], 1) * lam_vector[mask], 0) \
		+ (mu[n]**2 + beta[n]**2)

		# sample truncated normals
		key, key_next = jax.random.split(key)
		u = jax.random.uniform(key, [num_mc_samples, 2])
		mean, sdev = phi[n], jnp.diag(phi_cov[n])
		mc_samps = ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

		# monte carlo approximation of expectation
		mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], mc_samps[:, 0], mc_samps[:, 1]), 0)
		
		new_lam_vector = lam_vector.at[n].set(lam_mask * (I[n] > 0) * sigmoid(mcE - shape/(2 * rate) * arg))

		return new_lam_vector, key_next

	(lam, key_next) = fori_loop(0, N, body_fun, (lam, key))
	return lam, key_next

def _eval_lam_update_monte_carlo(I, phi_0, phi_1):
	fn = sigmoid(phi_0 * I - phi_1)
	return jnp.log(fn/(1 - fn))
_vmap_eval_lam_update_monte_carlo = jit(vmap(_eval_lam_update_monte_carlo, in_axes=(None, 0, 0)))

def _laplace_approx(y, phi_prior, phi_cov, I, key, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	''' Laplace approximation to sigmoid coefficient posteriors (phi).
	'''

	newton_steps = 10

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


#% Unused funcs that may be useful
@jit
def objective(y, u, v, lam):
    return (y - u @ v)**2 + lam * jnp.sum(v)

@jit
def objective_with_barrier(y, u, v, lam, t, mask):
    return (y - (u * mask) @ v)**2 + lam * jnp.sum(jnp.abs(v)) - 1/t * jnp.sum(jnp.log(v * (1 - v)))

@jit
def grad_fn(y, u, v, lam, t, mask):
    u_mask = u * mask
    return -2 * (y - u_mask @ v) * u_mask + lam - 1/t * (1 - 2*v)/(v * (1 - v))

@jit
def hess_fn(y, u, v, t, mask):
    return jnp.diag(2 * (u * mask)**2 + 1/t * (2 + (1 - 2*v)**2)/(v * (1 - v)))

@partial(jit, static_argnums=(6, 7, 8, 9))
def inner_newton(y, spks, mask, weights, lam, t, max_backtrack_iters, backtrack_alpha, backtrack_beta, eps=1e-5):
	step = 1
	J = grad_fn(y, weights, spks, lam, t, mask)
	H = hess_fn(y, weights, spks, t, mask)
	H_inv = jnp.diag(1/jnp.diag(H))
	search_dir = -H_inv @ J
	for bit in range(max_backtrack_iters):
		lhs = objective_with_barrier(y, weights, spks + step * search_dir, lam, t, mask)
		rhs = objective_with_barrier(y, weights, spks, lam, t, mask) + backtrack_alpha * step * J @ search_dir
		cond = jnp.min(jnp.array([(lhs > rhs) + jnp.isnan(lhs), 1])) # go condition
		step = step * backtrack_beta * cond + step * (1 - cond)
	spks += step * search_dir
	spks = jnp.where(spks > 1 - eps, 1 - eps, spks)
	spks = jnp.where(spks < eps, eps, spks)
	return spks

inner_newton_vmap = vmap(inner_newton, in_axes=(0, 1, 1, None, None, None, None, None, None))

def backtracking_newton_with_vmap(y, spks, tar_matrix, weights, lam_mask, newton_penalty=1e2, iters=20, barrier_iters=5, t=1e0, barrier_multiplier=1e1, 
						max_backtrack_iters=20, backtrack_alpha=0.05, backtrack_beta=0.75):
	
	for barrier_it in range(barrier_iters):
		for it in range(iters):
			spks = inner_newton_vmap(y, spks, tar_matrix, weights, newton_penalty, t, max_backtrack_iters, backtrack_alpha, backtrack_beta).T
		t *= barrier_multiplier

	return spks * lam_mask

def update_lam_backtracking_newton(y, lam, tar_matrix, mu, lam_mask, shape, rate, penalty=1, scale_factor=0.5, max_penalty_iters=10, 
	warm_start_lasso=False, constrain_weights='positive', barrier_iters=5, t=1e0, barrier_multiplier=1e1, max_backtrack_iters=20, 
	backtrack_alpha=0.05, backtrack_beta=0.75, verbose=False, tol=1e-3, newton_iters=20):
	
	N, K = lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	err_prev = 0

	if verbose:
			print(' ====== Updating lam via constrained L1 solver with iterative penalty shrinking ======')

	for it in range(max_penalty_iters):
		if verbose:
			print('penalty iter: ', it)
			print('current penalty: ', penalty)

		nlam = backtracking_newton_with_vmap(y, lam, tar_matrix, mu, lam_mask, newton_penalty=penalty, iters=newton_iters, barrier_iters=barrier_iters,
			t=t, barrier_multiplier=barrier_multiplier, max_backtrack_iters=max_backtrack_iters, backtrack_alpha=backtrack_alpha, backtrack_beta=backtrack_beta)

		err = np.sqrt(np.sum(np.square(y - mu @ nlam)))

		if verbose:
			print('newton err: ', err)
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

	return nlam
