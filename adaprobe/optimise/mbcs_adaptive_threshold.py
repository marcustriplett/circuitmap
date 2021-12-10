import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
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

def mbcs_adaptive_threshold(obs, I, mu_prior, beta_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, iters=50, 
	num_mc_samples=50, seed=0, y_xcorr_thresh=0.05, penalty=5e0, lam_masking=False, scale_factor=0.5, 
	max_penalty_iters=10, max_lasso_iters=100, warm_start_lasso=True, constrain_weights='positive', 
	verbose=False, learn_noise=False, init_lam=None, learn_lam=True, max_phi_thresh_iters=20, init_phi_thresh=0.2, 
	phi_thresh_scale_factor=0.95, min_phi_thresh=0.095, proportion_allowable_missed_events=0.1, phi_tol=1e-1, phi_delay=0, phi_thresh=0.09,
	outlier_penalty=10, orthogonal_outliers=True, minimum_spike_count=1, spont_rate=0., fit_excitability_intercept=True,
	assignment_threshold=0.2):
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
	z 			= np.zeros(K)
	
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
	shape_hist 		= jnp.zeros(iters)
	rate_hist 		= jnp.zeros(iters)
	phi_hist  		= jnp.zeros((iters, N, 2))
	phi_cov_hist 	= jnp.zeros((iters, N, 2, 2))
	z_hist 			= jnp.zeros((iters, K))
	
	hist_arrs = [mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist]

	# init key
	key = jax.random.PRNGKey(seed)

	# Iterate CAVI updates
	for it in tqdm(range(iters), desc='CAVI', leave=False):
		# print('iter %i/%i'%(it+1, iters), end='\r')
		beta = update_beta(lam, shape, rate, beta_prior)
		mu = update_mu_constr_l1(y - z, mu, lam, shape, rate, penalty=penalty, scale_factor=scale_factor, 
			max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, warm_start_lasso=warm_start_lasso, 
			constrain_weights=constrain_weights, verbose=verbose)
		lam, key = update_lam(y - z, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, key, num_mc_samples, N)
		if learn_noise:
			shape, rate = update_sigma(y, mu, beta, lam, shape_prior, rate_prior)
		mu, lam = collect_free_spikes(mu, lam, I, z, assignment_threshold=0.2)
		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)
		mu, lam = adaptive_excitability_threshold(mu, lam, I, phi, phi_thresh, minimum_spike_count=minimum_spike_count,
			spont_rate=spont_rate, fit_excitability_intercept=fit_excitability_intercept)
		if it > phi_delay:
			z = update_z_constr_l1(y, mu, lam, shape, rate, lam_mask, penalty=outlier_penalty, scale_factor=scale_factor,
				max_penalty_iters=max_penalty_iters, max_lasso_iters=max_lasso_iters, verbose=verbose, 
				orthogonal=orthogonal_outliers)

		# mu, lam = adaptive_excitability_threshold(y, mu, lam, phi, shape, rate, lam_mask, max_iters=max_phi_thresh_iters, 
		# 	init_thresh=init_phi_thresh, scale_factor=phi_thresh_scale_factor, min_thresh=min_phi_thresh, 
		# 	proportion_allowable_missed_events=proportion_allowable_missed_events, tol=phi_tol)

		# record history
		for hindx, pa in enumerate([mu, beta, lam, shape, rate, phi, phi_cov, z]):
			hist_arrs[hindx] = index_update(hist_arrs[hindx], it, pa)

	return mu, beta, lam, shape, rate, phi, phi_cov, z, *hist_arrs

def collect_free_spikes(mu, lam, I, z, assignment_threshold=0.2):
	powers = np.unique(I)
	for n in range(lam.shape[0]):
		locs = np.where(I[n] == powers[-1])[0]
		spont = np.where(z[locs])[0]
		if len(spont)/len(locs) >= assignment_threshold:
			# assign all spontaneous events to this cell
			locs_all = np.where(I[n] > 0)[0]
			spont_all = np.where(z[locs_all])[0]

			# what if cell n is disconnected?
			if mu[n] == 0:
				# reconnect n
				mu[n] = np.mean(z[locs_all[spont_all]])

			lam = index_update(lam, tuple([n, locs_all[spont_all]]), 1.)
			z[locs_all[spont_all]] = 0
	return lam

def adaptive_excitability_threshold(mu, lam, I, phi, phi_thresh, minimum_spike_count=1, spont_rate=0.1, fit_excitability_intercept=True):
	# Enforce monotonicity
	powers = np.unique(I)[1:]
	connected_cells = np.where(mu != 0)[0]
	n_connected = len(connected_cells)
	n_powers = len(powers)
	inferred_spk_probs = np.zeros((n_connected, n_powers))
	slopes = np.zeros(n_connected)
	# lr = LinearRegression(fit_intercept=fit_excitability_intercept)

	for i, n in enumerate(connected_cells):
		for p, power in enumerate(powers):
			locs = np.where(I[n] == power)[0]
			spks = np.where(lam[n, locs] >= 0.5)[0].shape[0]
			if locs.shape[0] > 0:
				inferred_spk_probs[i, p] = spks/locs.shape[0]
		slopes[i] = linregress(powers, inferred_spk_probs[i]).slope

	# disc_cells = connected_cells[slopes < 0]
	# mu = index_update(mu, disc_cells, 0.)
	# lam = index_update(lam, disc_cells, 0.)

	# Filter connection vector via opsin expression threshold
	phi_locs = np.where(phi[:, 0] < phi_thresh)[0]
	mu = index_update(mu, phi_locs, 0.)
	lam = index_update(lam, phi_locs, 0.)

	# Filter connection vector via spike counts
	spks = np.array([len(np.where(lam[n] >= 0.5)[0]) for n in range(mu.shape[0])])
	few_spk_locs = np.where(spks < minimum_spike_count)[0]
	mu = index_update(mu, few_spk_locs, 0.)
	lam = index_update(lam, few_spk_locs, 0.)

	return mu, lam

def _adaptive_excitability_threshold(y, mu, lam, phi, shape, rate, lam_mask, max_iters=20, init_thresh=0.2, scale_factor=0.95, min_thresh=0.09, 
	proportion_allowable_missed_events=0.1, tol=1e-1):
	'''Adaptively reduce excitability threshold phi until the L2 noise constraint is met
	'''
	sig = shape/rate
	phi_thresh = init_thresh
	mu_cpu = np.array(mu)
	lam_cpu = np.array(lam)
	y_cpu = np.array(y)
	y_pred = lam_cpu.T @ mu_cpu
	for it in range(max_iters):
		# Filter connection vector via opsin expression threshold
		phi_locs = phi[:, 0] < phi_thresh
		_mu = mu_cpu.copy()
		_mu[phi_locs] = 0
		_lam = lam_cpu.copy()
		_lam[phi_locs] = 0

		no_presynaptic_events = np.all(_lam[np.where(_mu != 0)[0]] < 0.5, axis=0)
		observed_events = np.where(lam_mask > 0)[0]

		# err = np.sum(no_presynaptic_events[observed_events])/(mu_cpu.shape[0] * observed_events.shape[0])
		err = np.sqrt(np.mean(np.square(y_pred - _lam.T @ _mu)))

		print('curr thresh: ', phi_thresh, ' err: ', err, ' constr: ', tol)
		if err <= tol or phi_thresh <= min_thresh:
			print('found excitability threshold ', phi_thresh)
			break
		else:
			phi_thresh *= scale_factor
	return _mu, _lam

@jit
def update_beta(lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(shape/rate * jnp.sum(lam, 1) + 1/(beta_prior**2))

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
	sigma = np.sqrt(rate/shape)
	constr = sigma * np.sqrt(K)
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
		if verbose:
			print('lasso err: ', err)
			print('constr: ', constr)
			print('')

	if constrain_weights == 'negative':
		return -coef
	else:
		return coef

def update_z_constr_l1(y, mu, Lam, shape, rate, lam_mask, penalty=1, scale_factor=0.5, max_penalty_iters=10, max_lasso_iters=100, 
	verbose=False, orthogonal=True):
	""" Soft thresholding with iterative penalty shrinkage
	"""
	if verbose:
		print(' ====== Updating z via soft thresholding with iterative penalty shrinking ======')

	N, K = Lam.shape
	sigma = np.sqrt(rate/shape)
	constr = sigma * np.sqrt(K)
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

# def _loss_fn(lam, args):
# 	y, w, lam_prior = args
# 	K, N = lam_prior.shape
# 	lam = lam.reshape([K, N])
# 	return np.sum(np.square(y - lam @ w)) - np.sum(lam * np.log(lam_prior) + (1 - lam) * np.log(1 - lam_prior))

# def _loss_fn_jax(lam, args):
# 	y, w, lam_prior = args
# 	K, N = lam_prior.shape
# 	lam = lam.reshape([K, N])
# 	return jnp.sum(jnp.square(y - lam @ w)) - jnp.sum(lam * jnp.log(lam_prior) + (1 - lam) * jnp.log(1 - lam_prior))

# _grad_loss_fn_jax = grad(_loss_fn_jax)
# _grad_loss_fn = lambda x, args: np.array(_grad_loss_fn_jax(x, args))

# def update_lam_bfgs(y, w, stim_matrix, phi, phi_cov, num_mc_samples=10):
# 	N, K = stim_matrix.shape
# 	unif_samples = np.random.uniform(0, 1, [N, 2, num_mc_samples])
# 	phi_samples = np.array([[ndtri(ndtr(-phi[n][i]/phi_cov[n][i,i]) + unif_samples[n, i] * (1 - ndtr(-phi[n][i]/phi_cov[n][i,i]))) * phi_cov[n][i,i] \
# 					  + phi[n][i] for i in range(2)] for n in range(N)])
# 	lam_prior = np.mean([sigmoid(phi_samples[:, 0, i][:, None] * stim_matrix - phi_samples[:, 1, i][:, None]) for i in range(num_mc_samples)], axis=0)
# 	args = [y, w, lam_prior.T]
# 	res = minimize(_loss_fn, lam_prior.T.flatten(), jac=_grad_loss_fn, args=args, method='L-BFGS-B', bounds=[(0, 1)]*(K*N))
# 	return res.x.reshape([K, N]).T

@jax.partial(jit, static_argnums=(11, 12, 13)) # lam_mask[k] = 1 if xcorr(y_psc[k]) > thresh else 0.
def update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, key, num_mc_samples, N):
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
def update_sigma(y, mu, beta, lam, shape_prior, rate_prior):
	K = y.shape[0]
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (jnp.sum(jnp.square(y - np.sum(jnp.expand_dims(mu, 1) * lam, 0))) \
		- jnp.sum(jnp.square(jnp.expand_dims(mu, 1) * lam)) + jnp.sum(jnp.expand_dims((mu**2 + beta**2), 1) * lam))
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
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) - jnp.sum(jnp.log(phi))/t \
		+ 1/2 * (phi - phi_prior) @ prec @ (phi - phi_prior)
