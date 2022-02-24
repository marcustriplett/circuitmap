import numpy as np
# from sklearn.isotonic import IsotonicRegression


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
from tqdm import trange

from .pava import _isotonic_regression, simultaneous_isotonic_regression

EPS = 1e-10

def cavi_sns(y_psc, I, mu_prior, beta_prior, alpha_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	iters, num_mc_samples, seed, y_xcorr_thresh=1e-2, learn_noise=False, phi_thresh=None, minimum_spike_count=3,
	phi_thresh_delay=1, minimax_spk_prob=0.3, scale_factor=0.75, penalty=2e1, lam_iters=1, disc_strength=0.05,
	noise_scale=0.5, noise_update='iid'):
	y = np.trapz(y_psc, axis=-1)
	K = y.shape[0]
	lam_mask = jnp.array([jnp.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh

	lam = np.zeros_like(I)
	lam[I > 0] = 0.95
	lam = lam * lam_mask

	return _cavi_sns(y, I, mu_prior, beta_prior, alpha_prior, lam, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	lam_mask, iters, num_mc_samples, seed, learn_noise, phi_thresh, phi_thresh_delay, minimax_spk_prob, minimum_spike_count, 
	scale_factor, penalty, lam_iters, disc_strength, noise_scale, noise_update)

# @jax.partial(jit, static_argnums=(10, 11, 12, 13, 14, 15))
def _cavi_sns(y, I, mu_prior, beta_prior, alpha_prior, lam, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	lam_mask, iters, num_mc_samples, seed, learn_noise, phi_thresh, phi_thresh_delay, minimax_spk_prob, minimum_spike_count,
	scale_factor, penalty, lam_iters, disc_strength, noise_scale, noise_update):
	"""Offline-mode coordinate ascent variational inference for the adaprobe model.
	"""

	spont_rate = 0.

	# Initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]

	# Declare scope types
	mu 			= jnp.array(mu_prior)
	beta 		= jnp.array(beta_prior)
	alpha 		= jnp.array(alpha_prior)
	shape 		= jnp.array(shape_prior)
	rate 		= jnp.array(rate_prior)
	phi 		= jnp.array(phi_prior)
	phi_cov 	= jnp.array(phi_cov_prior)
	z 			= np.zeros(K)
	rfs 		= None # prevent error when num-iters < phi_thresh_delay

	# Define history arrays
	cpus = jax.devices('cpu')

	mu_hist 	= jax.device_put(np.zeros((iters, N)), cpus[0])
	beta_hist 	= jax.device_put(np.zeros((iters, N)), cpus[0])
	alpha_hist 	= jax.device_put(np.zeros((iters, N)), cpus[0])
	lam_hist 	= jax.device_put(np.zeros((iters, N, K)), cpus[0])
	shape_hist 	= jax.device_put(np.zeros((iters, K)), cpus[0])
	rate_hist 	= jax.device_put(np.zeros((iters, K)), cpus[0])
	phi_hist  	= jax.device_put(np.zeros((iters, N, 2)), cpus[0])
	phi_cov_hist = jax.device_put(np.zeros((iters, N, 2, 2)), cpus[0])
	z_hist 		= jax.device_put(np.zeros((iters, K)), cpus[0])
	
	hist_arrs = [mu_hist, beta_hist, alpha_hist, lam_hist, shape_hist, rate_hist, \
		phi_hist, phi_cov_hist, z_hist]

	# init key
	key = jax.random.PRNGKey(seed)

	# Iterate CAVI updates
	for it in trange(iters):
		beta = update_beta(alpha, lam, shape, rate, beta_prior)
		mu, key = update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, N, key)
		alpha, key = update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, N, key)
		for _ in range(lam_iters):
			lam, key = update_lam(y, I, mu, beta, alpha, lam, shape, rate, \
				phi, phi_cov, lam_mask, key, num_mc_samples, N)

		if noise_update == 'iid':
			shape, rate = update_sigma(y, mu, beta, alpha, lam, shape_prior, rate_prior)
		elif noise_update == 'trial-wise':
			shape, rate, key = update_noise(y, mu, beta, alpha, lam, key, noise_scale=noise_scale, num_mc_samples=num_mc_samples)
		else:
			raise Exception

		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)

		if it > phi_thresh_delay:
			rfs, disc_cells = update_isotonic_receptive_field(lam, I, minimax_spk_prob=minimax_spk_prob + spont_rate, minimum_spike_count=minimum_spike_count)
			for n in range(N):
				alpha = index_update(alpha, n, alpha[n] * (1. - disc_cells[n]) + disc_cells[n] * disc_strength) # strongly believes cell is disconnected
				mu = index_update(mu, n, mu[n] * (1. - disc_cells[n]) + disc_cells[n] * disc_strength)
				lam = index_update(lam, n, lam[n] * (1. - disc_cells[n]) + disc_cells[n] * 0.)
			z = update_z_l1_with_residual_tolerance(y, alpha, mu, lam, lam_mask, scale_factor=scale_factor, penalty=penalty)
			spont_rate = np.mean(z != 0.)


		for hindx, pa in enumerate([mu, beta, alpha, lam, shape, rate, phi, phi_cov, z]):
			hist_arrs[hindx] = index_update(hist_arrs[hindx], it, pa)

	mu, beta, alpha, lam, z = reconnect_spont_cells(y, I, lam, mu, alpha, beta, z, minimax_spk_prob=minimax_spk_prob)
	(phi, phi_cov), _ = update_phi(lam, I, phi_prior, phi_cov_prior, key)

	return mu, beta, alpha, lam, shape, rate, phi, phi_cov, z, rfs, *hist_arrs

def reconnect_spont_cells(y, stim_matrix, lam, mu, alpha, beta, z, minimax_spk_prob=0.3):
	disc_cells = np.where(mu == 0.)[0]
	powers = np.unique(stim_matrix)[1:] # skip zero power
	z = np.array(z)
	
	print('Examining %i cells for false negatives...'%len(disc_cells))
	while len(disc_cells) > 0:
		stim_locs = []
		for n in disc_cells:
			stim_locs += [np.where(z[np.where(stim_matrix[n])[0]])[0]]

		# Focus on cell with largest number of associated spikes
		focus_indx = np.argmax([len(sl) for sl in stim_locs])
		focus = disc_cells[focus_indx]

		# Check pava condition
		srates = np.zeros_like(powers)
		for i, p in enumerate(powers):
			srates[i] = np.mean(z[np.where(stim_matrix[focus] == p)[0]] != 0)
		pava = _isotonic_regression(srates, np.ones_like(srates))[-1]
		
		if pava >= minimax_spk_prob:
			# Passes pava condition, reconnect cell
			print('Reconnecting cell %i with maximal pava spike rate %.2f'%(focus, pava))
			z_locs = np.intersect1d(np.where(stim_matrix[focus])[0], np.where(z)[0])
			mu = index_update(mu, focus, np.mean(z[z_locs]))
			beta = index_update(beta, focus, np.std(z[z_locs]))
			alpha = index_update(alpha, focus, 1.)
			lam = index_update(lam, (focus, z_locs), 1.)
			z[z_locs] = 0. # delete events from spont vector

		disc_cells = np.delete(disc_cells, focus_indx)
		print('deleting cell %i from reconnection candidates'%focus)

	print('Cell reconnection complete.')

	return mu, beta, alpha, lam, z # then update phi


@jax.partial(jit, static_argnums=(7))
def update_noise(y, mu, beta, alpha, lam, key, noise_scale=0.5, num_mc_samples=10):
	N, K = lam.shape
	std = beta * (mu != 0)

	alpha_samps = (jax.random.uniform(key, [num_mc_samples, N]) <= alpha) * 1.0
	key, _ = jax.random.split(key)

	w_samps = (mu + std * jax.random.normal(key, [num_mc_samples, N])) * alpha_samps
	key, _ = jax.random.split(key)
	# w_samps = np.random.normal(mu, std, [num_mc_samples, N])

	s_samps = (jax.random.uniform(key, [num_mc_samples, N, K]) <= lam) * 1.0
	key, _ = jax.random.split(key)
	# s_samps = (np.random.rand(num_mc_samples, N, K) <= lam[None, :, :]).astype(float)

	mc_ws_sq = jnp.mean(jnp.sum((w_samps[..., jnp.newaxis] * s_samps)**2, axis=1), axis=0)
	# mc_ws_sq = np.mean([(w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0)

	mc_recon_err = jnp.mean((y - jnp.sum(w_samps[..., jnp.newaxis] * s_samps, axis=1))**2, axis=0)
	# mc_recon_err = np.mean([(y - w_samps[i] @ s_samps[i])**2 for i in range(num_mc_samples)], axis=0)

	shape = noise_scale**2 * mc_ws_sq + 1/2
	rate = noise_scale * (mu * alpha) @ lam + 1/2 * mc_recon_err + 1e-5
	return shape, rate, key

def update_isotonic_receptive_field(lam, stim_matrix, minimax_spk_prob=0.3, minimum_spike_count=3):
	N, K = lam.shape
	# lam = np.array(_lam) # convert to ndarray
	powers = np.unique(stim_matrix)[1:] # discard zero
	n_powers = len(powers)
	inferred_spk_probs = jnp.zeros((N, n_powers))
	# isotonic_regressor = IsotonicRegression(y_min=0, y_max=1, increasing=True)
	disc_cells = np.zeros(N)
	# receptive_field = jnp.zeros((N, n_powers))
	jones = jnp.ones(n_powers)

	for n in range(N):
		for p, power in enumerate(powers):
			locs = np.where(stim_matrix[n] == power)[0]
			if locs.shape[0] > 0:
				inferred_spk_probs = index_update(inferred_spk_probs, (n, p + 1), jnp.mean(lam[n, locs]))

		# isotonic_regressor.fit(powers, inferred_spk_probs[n])
		# receptive_field[n] = isotonic_regressor.f_(powers)

	# receptive_field = index_update(receptive_field, n, _isotonic_regression(inferred_spk_probs[n], jones))
	receptive_field = simultaneous_isotonic_regression(powers, inferred_spk_probs)

	# if receptive_field[n, -1] < minimax_spk_prob or jnp.sum(lam[n]) < minimum_spike_count:
	disc_locs = np.unique(np.concatenate([np.where(receptive_field[:, -1] < minimax_spk_prob)[0], np.where(jnp.sum(lam, axis=1))[0]]))
	disc_cells[disc_locs] = 1.

	return receptive_field, disc_cells

def update_z_l1_with_residual_tolerance(y, _alpha, _mu, _lam, lam_mask, penalty=2e1, scale_factor=0.5, max_penalty_iters=50, verbose=False, 
	orthogonal=True, tol=0.05):
	""" Soft thresholding with iterative penalty shrinkage
	"""
	if verbose:
		print(' ==== Updating z via soft thresholding with iterative penalty shrinking ==== ')

	alpha, mu, lam = np.array(_alpha), np.array(_mu), np.array(_lam)
	N, K = lam.shape
	resid = np.array(y - lam.T @ (mu * alpha))

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

@jit
def update_beta(alpha, lam, shape, rate, beta_prior):
	return 1/jnp.sqrt(alpha * jnp.sum(shape/rate * lam, 1) + 1/(beta_prior**2))

@jax.partial(jit, static_argnums=(9))
def update_mu(y, mu, beta, alpha, lam, shape, rate, mu_prior, beta_prior, N, key):
	"""Update based on solving E_q(Z-mu_n)[ln p(y, Z)]"""
	sig = shape/rate
	update_order = jax.random.choice(key, N, [N], replace=False)
	with loops.Scope() as scope:
		scope.mu = mu
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.all_ids = jnp.arange(N)
		for m in scope.range(N):
			n = update_order[m]
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, jnp.mod(n - 1, N)), size=N-1)
			scope.mu = index_update(scope.mu, n, (beta[n]**2) * (alpha[n] * jnp.dot(sig * y, lam[n]) - alpha[n] \
				* jnp.dot(sig * lam[n], jnp.sum(jnp.expand_dims(scope.mu[scope.mask] * alpha[scope.mask], 1) * lam[scope.mask], 0)) \
				+ mu_prior[n]/(beta_prior[n]**2)))
	key, _ = jax.random.split(key)
	return scope.mu, key

@jax.partial(jit, static_argnums=(8))
def update_alpha(y, mu, beta, alpha, lam, shape, rate, alpha_prior, N, key):
	update_order = jax.random.choice(key, N, [N], replace=False)
	sig = shape/rate
	with loops.Scope() as scope:
		scope.alpha = alpha
		scope.arg = 0.
		scope.mask = jnp.zeros(N - 1, dtype=int)
		scope.all_ids = jnp.arange(N)
		for m in scope.range(N):
			n = update_order[m]
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, jnp.mod(n - 1, N)), size=N-1) 
			scope.arg = -2 * mu[n] * jnp.dot(sig * y, lam[n]) + 2 * mu[n] * jnp.dot(sig * lam[n], jnp.sum(jnp.expand_dims(mu[scope.mask] * scope.alpha[scope.mask], 1) \
				* lam[scope.mask], 0)) + (mu[n]**2 + beta[n]**2) * jnp.sum(sig * lam[n])
			scope.alpha = index_update(scope.alpha, n, sigmoid(jnp.log((alpha_prior[n] + EPS)/(1 - alpha_prior[n] + EPS)) - 1/2 * scope.arg))
	key, _ = jax.random.split(key)
	return scope.alpha, key

@jax.partial(jit, static_argnums=(12, 13)) # lam_mask[k] = 1 if xcorr(y_psc[k]) > thresh else 0.
def update_lam(y, I, mu, beta, alpha, lam, shape, rate, phi, phi_cov, lam_mask, key, num_mc_samples, N):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	K = I.shape[1]
	update_order = jax.random.choice(key, N, [N], replace=False)
	sig = shape/rate
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

		for m in scope.range(N):
			n = update_order[m]
			scope.mask = jnp.unique(jnp.where(scope.all_ids != n, scope.all_ids, jnp.mod(n - 1, N)), size=N-1)
			scope.arg = -2 * sig * y * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * jnp.sum(sig * jnp.expand_dims(mu[scope.mask] * alpha[scope.mask], 1) * scope.lam[scope.mask], 0) \
			+ sig * (mu[n]**2 + beta[n]**2) * alpha[n]

			# sample truncated normals
			scope.key, scope.key_next = jax.random.split(scope.key)
			scope.u = jax.random.uniform(scope.key, [num_mc_samples, 2])
			scope.mean, scope.sdev = phi[n], jnp.diag(phi_cov[n])
			scope.mc_samps = ndtri(ndtr(-scope.mean/scope.sdev) + scope.u * (1 - ndtr(-scope.mean/scope.sdev))) * scope.sdev + scope.mean

			# monte carlo approximation of expectation
			scope.mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], scope.mc_samps[:, 0], scope.mc_samps[:, 1]), 0)
			scope.lam = index_update(scope.lam, n, lam_mask * (I[n] > 0) * sigmoid(scope.mcE - 1/2 * scope.arg)) # require spiking cells to be targeted
			# scope.lam = index_update(scope.lam, n, lam_mask * (I[n] > 0) * (mu[n] != 0) * sigmoid(scope.mcE - shape/(2 * rate) * scope.arg)) # require spiking cells to be targeted
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
