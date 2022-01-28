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

def mbcs_spike_weighted_var_with_outliers(y_psc, I, mu_prior, beta_prior, shape_prior, rate_prior, phi_prior, 
	phi_cov_prior, iters=50, num_mc_samples=50, seed=0, y_xcorr_thresh=0.05, penalty=5e0, scale_factor=0.5,
	max_penalty_iters=10, max_lasso_iters=100, warm_start_lasso=True, constrain_weights='positive',
	verbose=False, learn_noise=False, init_lam=None, learn_lam=True, phi_delay=-1, phi_thresh=0.09,
	minimum_spike_count=1, noise_scale=0.5, num_mc_samples_noise_model=10, minimum_maximal_spike_prob=0.2, 
	orthogonal_outliers=True, outlier_penalty=5e1, init_spike_prior=0.75, outlier_tol=0.05, spont_rate=0,
	lam_mask_fraction=0.05):
	"""Offline-mode coordinate ascent variational inference for the adaprobe model.
	"""

	y = np.trapz(y_psc, axis=-1)
	K = y.shape[0]

	# Setup lam mask
	lam_mask = (np.array([np.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh)
	lam_mask[np.max(y_psc, axis=1) < lam_mask_fraction * np.max(y_psc)] = 0 # mask events that are too small

	# Initialise new params
	N = mu_prior.shape[0]

	# Declare scope types
	mu 			= jnp.array(mu_prior)
	beta 		= jnp.array(beta_prior)
	shape 		= shape_prior
	rate 		= rate_prior
	phi 		= jnp.array(phi_prior)
	phi_cov 	= jnp.array(phi_cov_prior)
	z 			= np.zeros(K)
	
	# Spike initialisation
	lam = np.zeros_like(I) 
	lam[I > 0] = 0.95
	lam = lam * lam_mask

	lam = jnp.array(lam) # move to DeviceArray

	# Define history arrays
	mu_hist 		= jnp.zeros((iters, N))
	beta_hist 		= jnp.zeros((iters, N))
	lam_hist 		= jnp.zeros((iters, N, K))
	shape_hist 		= jnp.zeros((iters, K))
	rate_hist 		= jnp.zeros((iters, K))
	phi_hist  		= jnp.zeros((iters, N, 2))
	phi_cov_hist 	= jnp.zeros((iters, N, 2, 2))
	z_hist 			= jnp.zeros((iters, K))

	hist_arrs = [mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist]

	# init key
	key = jax.random.PRNGKey(seed)

	# init spike prior
	spike_prior = np.zeros_like(lam)
	spike_prior[lam > 0] = init_spike_prior

	# Iterate CAVI updates
	for it in tqdm(range(iters), desc='CAVI', leave=True):
		beta = update_beta(lam, shape, rate, beta_prior)
		# ignore z during mu and lam updates
		mu, lam = update_mu_constr_l1(y, mu, lam, shape, rate, penalty=penalty, scale_factor=scale_factor, 
			max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, warm_start_lasso=warm_start_lasso, 
			constrain_weights=constrain_weights, verbose=verbose)
		update_order = np.random.choice(N, N, replace=False)
		lam = update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N)
		receptive_field, spike_prior = update_isotonic_receptive_field(lam, I)
		mu, lam = isotonic_filtering(mu, lam, I, receptive_field, minimum_spike_count=minimum_spike_count, minimum_maximal_spike_prob=minimum_maximal_spike_prob + spont_rate)
		shape, rate = update_noise(y, mu, beta, lam, noise_scale=noise_scale, num_mc_samples=num_mc_samples_noise_model)

		if it > phi_delay:
			z = update_z_l1_with_residual_tolerance(y, mu, lam, lam_mask, penalty=outlier_penalty, scale_factor=scale_factor,
				max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, verbose=verbose, 
				orthogonal=orthogonal_outliers, tol=outlier_tol)
			spont_rate = np.mean(z != 0)

		# record history
		for hindx, pa in enumerate([mu, beta, lam, shape, rate, phi, phi_cov, z]):
			hist_arrs[hindx] = index_update(hist_arrs[hindx], it, pa)

	return mu, beta, lam, shape, rate, phi, phi_cov, z, *hist_arrs

def update_noise(y, mu, beta, lam, noise_scale=0.5, num_mc_samples=10, eps=1e-4):
	N, K = lam.shape
	std = beta * (mu != 0)
	w_samps = np.random.normal(mu, std, [num_mc_samples, N])
	s_samps = (np.random.rand(num_mc_samples, N, K) <= lam[None, :, :]).astype(float)
	mc_ws_sq = np.mean([(w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0)
	mc_recon_err = np.mean([(y - w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0) # multiply by s_samps here to ensure zero noise on no-spike trials
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

def update_mu_constr_l1(y, mu, Lam, shape, rate, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, \
	warm_start_lasso=False, constrain_weights='positive', verbose=False, tol=1e-5):
	""" Constrained L1 solver with iterative penalty shrinking
	"""
	if verbose:
		print(' ====== Updating mu via constrained L1 solver with iterative penalty shrinking ======')
	N, K = Lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	mu, Lam = np.array(mu), np.array(Lam) # cast to ndarray

	LamT = Lam.T
	positive = constrain_weights in ['positive', 'negative']
	lasso = Lasso(alpha=penalty, fit_intercept=False, max_iter=max_lasso_iters, warm_start=warm_start_lasso, positive=positive)
	
	if constrain_weights == 'negative':
		# make sensing matrix and weight warm-start negative
		LamT = -LamT
		mu = -mu
	lasso.coef_ = mu

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

	# zero-out spikes for disconnected cells
	Lam[np.where(mu) == 0] = 0

	if constrain_weights == 'negative':
		return -coef, Lam
	else:
		return coef, Lam

def update_z_l1_with_residual_tolerance(y, mu, Lam, lam_mask, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, 
	verbose=False, orthogonal=True, tol=0.05):
	""" Soft thresholding with iterative penalty shrinkage
	"""
	if verbose:
		print(' ====== Updating z via soft thresholding with iterative penalty shrinking ======')

	N, K = Lam.shape
	resid = np.array(y - Lam.T @ mu) # copy to np array, possible memory overhead problem here

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
			z[np.any(Lam >= 0.5, axis=0)] = 0

		z = z * lam_mask
		err = np.sum(np.square(resid - z))/np.sum(np.square(y))

		if verbose:
			print('soft thresh err: ', err)
			print('tol: ', tol)
			print('')

		if err <= tol:
			if verbose:
				print(' ==== converged on iteration: %i ===='%it)
			break
		else:
			penalty *= scale_factor # exponential backoff
		
	return z

def update_z_constr_l1(y, mu, Lam, shape, rate, lam_mask, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, 
	verbose=False, orthogonal=True):
	""" Soft thresholding with iterative penalty shrinkage
	"""
	if verbose:
		print(' ====== Updating z via soft thresholding with iterative penalty shrinking ======')

	N, K = Lam.shape
	constr = np.sqrt(np.sum(rate/shape))
	resid = np.array(y - Lam.T @ mu) # copy to np array, possible memory overhead problem here

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
			z[np.any(Lam >= 0.5, axis=0)] = 0

		z = z * lam_mask
		err = np.sqrt(np.sum(np.square(resid - z)))

		if verbose:
			print('soft thresh err: ', err)
			print('constr: ', constr)
			print('')

		if err <= constr:
			if verbose:
				print(' ==== converged on iteration: %i ===='%it)
			break
		else:
			penalty *= scale_factor # exponential backoff
		
	return z

def update_lam_with_isotonic_receptive_field(y, I, mu, beta, lam, shape, rate, lam_mask, update_order, spike_prior, num_mc_samples, N):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""

	K = I.shape[1]
	all_ids = jnp.arange(N)

	for m in range(N):
		n = update_order[m]
		if mu[n] != 0:
			# mask = jnp.unique(jnp.where(all_ids != n, all_ids, n - 1), size=N-1)
			mask = jnp.array(np.delete(all_ids, n)).squeeze()
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

