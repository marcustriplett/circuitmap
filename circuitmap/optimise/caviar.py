import numpy as np
from scipy.stats import sem

# Jax imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, while_loop, fori_loop
from jax.nn import sigmoid
from jax.scipy.special import ndtr, ndtri

from jax import config; config.update("jax_enable_x64", True)
from functools import partial

# Experimental loops
from tqdm import trange

from .pava import _isotonic_regression

def caviar(y_psc, I, mu_prior, beta_prior, shape_prior, rate_prior, phi_prior, phi_cov_prior, 
	iters=50, num_mc_samples=100, seed=0, y_xcorr_thresh=1e-2, minimum_spike_count=3,
	delay_spont_est=1, msrmp=0.3, scale_factor=0.75, penalty=5e0, save_histories=False, 
	max_backtrack_iters=20, tol=0.05, spont_orthogonality=0.1, fn_scan=True):
	''' Coordinate-ascent variational inference and isotonic regularisation.
	'''
	print('Running coordinate-ascent variational inference and isotonic regularisation (CAVIaR) algorithm.')
	
	y = np.trapz(y_psc, axis=-1)
	K = y.shape[0]
	lam_mask = jnp.array([jnp.correlate(y_psc[k], y_psc[k]) for k in range(K)]).squeeze() > y_xcorr_thresh

	lam = np.zeros_like(I)
	lam[I > 0] = 0.95
	lam = lam * lam_mask
	I = jnp.array(I.astype(float))

	spont_rate = 0.

	# initialise new params
	N = mu_prior.shape[0]
	K = y.shape[0]
	powers = jnp.array(np.unique(I)[1:])

	mu = jnp.array(mu_prior)
	beta = jnp.array(beta_prior)
	shape = jnp.array(shape_prior)
	rate = jnp.array(rate_prior)
	phi = jnp.array(phi_prior)
	phi_cov = jnp.array(phi_cov_prior)
	z = jnp.zeros(K)
	receptive_fields = None # prevent error when num-iters < phi_thresh_delay

	if save_histories:
		cpu = jax.devices('cpu')[0]

		# define history arrays
		mu_hist = np.zeros((iters, N))
		beta_hist = np.zeros((iters, N))
		lam_hist = np.zeros((iters, N, K))
		shape_hist = np.zeros((iters, K))
		rate_hist = np.zeros((iters, K))
		phi_hist = np.zeros((iters, N, 2))
		phi_cov_hist = np.zeros((iters, N, 2, 2))
		z_hist = np.zeros((iters, K))
		
		hist_arrs = [mu_hist, beta_hist, lam_hist, shape_hist, rate_hist, \
			phi_hist, phi_cov_hist, z_hist]

		# move hist arrays to CPU
		hist_arrs = [jax.device_put(ha, cpu) for ha in hist_arrs]

	else:
		hist_arrs = [None] * 8
		
	# init key
	key = jax.random.PRNGKey(seed)

	# iterate CAVIaR updates
	for it in trange(iters):
		mu, beta = block_update_mu(y, mu, beta, lam, shape, rate, mu_prior, beta_prior, N)
		lam, key = update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, key, 
								num_mc_samples, N, powers, minimum_spike_count, msrmp + spont_rate, 
								it, delay_spont_est)
		shape, rate = update_sigma(y, mu, beta, lam, shape_prior, rate_prior)
		(phi, phi_cov), key = update_phi(lam, I, phi_prior, phi_cov_prior, key)
		z = estimate_spont_act_soft_thresh((y, mu, lam, it, max_backtrack_iters, jnp.sum(y), 
								z, penalty, lam_mask, scale_factor, tol, spont_orthogonality))[6]
		spont_rate = jnp.mean(z != 0.)

		if save_histories:
			for hindx, pa in enumerate([mu, beta, lam, shape, rate, phi, phi_cov, z]):
				hist_arrs[hindx] = hist_arrs[hindx].at[it].set(pa)

	if fn_scan:
		# final scan for possible false negatives
		mu, beta, lam, z = reconnect_spont_cells(y, I, lam, mu, beta, z, minimax_spk_prob=msrmp, 
			minimum_spike_count=minimum_spike_count)
		(phi, phi_cov), _ = update_phi(lam, I, phi_prior, phi_cov_prior, key)

	return (mu, beta, lam, shape, rate, phi, phi_cov, z, receptive_fields, *hist_arrs)

def reconnect_spont_cells(y, stim_matrix, lam, mu, beta, z, minimax_spk_prob=0.3, minimum_spike_count=3):
	disc_cells = np.where(mu == 0.)[0]
	powers = np.unique(stim_matrix)[1:] # skip zero power
	z = np.array(z) # convert to np array
	
	init_num_disc_cells = len(disc_cells)
	print('Examining %i/%i cells for false negatives...'%(len(disc_cells), stim_matrix.shape[0]))
	while len(disc_cells) > 0:
		print('Remaining cells: %i/%i'%(len(disc_cells), init_num_disc_cells), end='\r')
		if len(np.where(z)[0]) > minimum_spike_count:
			stim_locs = []
			for n in disc_cells:
				stim_locs += [np.where(z[np.where(stim_matrix[n])[0]])[0]]

			# Focus on cell with largest number of associated spikes
			focus_indx = np.argmax([len(sl) for sl in stim_locs])
			focus = disc_cells[focus_indx]

			# Check pava condition
			srates = np.zeros_like(powers)
			spike_count = 0
			for i, p in enumerate(powers):
				z_locs = np.where(stim_matrix[focus] == p)[0]
				if len(z_locs) > 0:
					srates[i] = np.mean(z[z_locs] != 0)
					spike_count += np.sum(z[z_locs] != 0)
			pava = _isotonic_regression(srates)[-1]
			
			if pava >= minimax_spk_prob and spike_count >= minimum_spike_count:
				# Passes pava condition, reconnect cell
				print('Reconnecting cell %i with maximal pava spike rate %.2f'%(focus, pava))
				z_locs = np.intersect1d(np.where(stim_matrix[focus])[0], np.where(z)[0])
				mu = mu.at[focus].set(np.mean(z[z_locs]))
				beta = beta.at[focus].set(sem(z[z_locs]))
				lam = lam.at[(focus, z_locs)].set(1.)
				z[z_locs] = 0. # delete events from spont vector

			disc_cells = np.delete(disc_cells, focus_indx)
		else:
			break

	print('Cell reconnection complete.')
	return mu, beta, lam, z

#% estimate_spont_act_soft_thresh
def _esast_cond_fun(carry):
	it, max_iters, err, tol = [carry[i] for i in [3, 4, 5, 10]]
	return jnp.logical_and(it < max_iters, err > tol)

def _esast_body_fun(carry):
	y, mu, lam, it, max_iters, err, z, pen, mask, scale_factor, tol, spont_orthogonality = carry
	resid = y - lam.T @ mu
	z = jnp.where(resid < pen, 0., resid - pen)
	z = jnp.where(z < 0., 0., z)
	z = jnp.where(jnp.any(lam >= spont_orthogonality, axis=0), 0., z)
	z *= mask
	err = jnp.sum(jnp.square(resid - z))/(jnp.sum(jnp.square(y)) + 1e-5)
	it += 1
	pen *= scale_factor
	return y, mu, lam, it, max_iters, err, z, pen, mask, scale_factor, tol, spont_orthogonality

estimate_spont_act_soft_thresh = jit(lambda carry: while_loop(_esast_cond_fun, _esast_body_fun, carry))


@partial(jit, static_argnums=(8))
def block_update_mu(y, mu, beta, lam, shape, rate, mu_prior, beta_prior, N):
	L = lam @ lam.T
	D = jnp.diag(jnp.sum(lam * (1 - lam), axis=-1))
	posterior_cov = jnp.linalg.inv(shape/rate * (D + L) + 1/(beta_prior**2) * jnp.eye(N))
	posterior_mean = posterior_cov @ (shape/rate * jnp.sum(y * lam, axis=1) + 1/(beta_prior**2) * mu_prior)
	return posterior_mean, jnp.diag(posterior_cov)

def _eval_spike_rates(stimv, lamv, powers):
	K = stimv.shape[0]
	Krange = jnp.arange(K)
	npowers = powers.shape[0]

	def body_fun(p, inf_spike_rates):
		power = powers[p]
		locs = jnp.where(stimv == power, Krange, -1)
		mask = (locs >= 0)
		sr = jnp.sum(lamv[locs] * mask)/(jnp.sum(mask) + 1e-4 * (jnp.sum(mask) == 0.))
		return inf_spike_rates.at[p].set(sr)

	return fori_loop(0, npowers, body_fun, jnp.zeros(npowers))

eval_spike_rates = vmap(_eval_spike_rates, in_axes=(0, 0, None))

@partial(jit, static_argnums=(11, 12)) # lam_mask[k] = 1 if xcorr(y_psc[k]) > thresh else 0.
def update_lam(y, I, mu, beta, lam, shape, rate, phi, phi_cov, lam_mask, key, num_mc_samples, N, powers, 
	minimum_spike_count, minimax_spk_prob, it, delay_spont_est):
	'''Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	'''
	K = I.shape[1]
	update_order = jax.random.choice(key, N, [N], replace=False)
	sig = shape/rate
	all_ids = jnp.arange(N)
		
	def body_fun(m, carry):
		lam, mu, key = carry

		n = update_order[m]
		mask = jnp.unique(jnp.where(all_ids != n, all_ids, jnp.mod(n - 1, N)), size=N-1)
		arg = -2 * sig * y * mu[n] + 2 * mu[n] * jnp.sum(sig * jnp.expand_dims(mu[mask], 1) * lam[mask], 0) \
			+ sig * (mu[n]**2 + beta[n]**2)

		# sample truncated normals
		key, key_next = jax.random.split(key)
		u = jax.random.uniform(key, [num_mc_samples, 2])
		mean, sdev = phi[n], jnp.diag(phi_cov[n])
		mc_samps = ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

		# monte carlo approximation of expectation
		mcE = jnp.mean(_vmap_eval_lam_update_monte_carlo(I[n], mc_samps[:, 0], mc_samps[:, 1]), 0)
		est_lam = lam_mask * (I[n] > 0) * sigmoid(mcE - 1/2 * arg) # require spiking cells to be targeted

		# check pava condition
		srates = _eval_spike_rates(I[n], est_lam, powers)
		pava = (_isotonic_regression(srates)[-1] >= minimax_spk_prob) * (jnp.sum(est_lam) >= minimum_spike_count)
		pava = pava * (it > delay_spont_est) + 1. * (it <= delay_spont_est)

		# update lam
		new_lam = lam.at[n].set(est_lam * pava)
		new_mu = mu.at[n].set(mu[n] * (pava == 1.))

		return (new_lam, new_mu, key_next)

	lam, mu, key = fori_loop(0, N, body_fun, (lam, mu, key))

	return lam, key

def _eval_lam_update_monte_carlo(I, phi_0, phi_1):
	fn = sigmoid(phi_0 * I - phi_1)
	return jnp.log(fn/(1 - fn))
_vmap_eval_lam_update_monte_carlo = jit(vmap(_eval_lam_update_monte_carlo, in_axes=(None, 0, 0)))

@jit
def update_sigma(y, mu, beta, lam, shape_prior, rate_prior):
	K = y.shape[0]
	shape = shape_prior + K/2
	rate = rate_prior + 1/2 * (jnp.sum(jnp.square(y - np.sum(jnp.expand_dims(mu, 1) * lam, 0))) \
		- jnp.sum(jnp.square(jnp.expand_dims(mu, 1) * lam)) + jnp.sum(jnp.expand_dims(mu**2 + beta**2, 1) * lam))
	return shape, rate

@jit
def update_phi(lam, I, phi_prior, phi_cov_prior, key):
	''' Returns updated sigmoid coefficients estimated using a log-barrier penalty with backtracking Newton's method
	'''
	(posterior, logliks), keys = laplace_approx(lam, phi_prior, phi_cov_prior, I, key) # N keys returned due to vmapped LAs
	return posterior, keys[-1]
	
def _laplace_approx(y, phi_prior, phi_cov, I, key, t=1e1, backtrack_alpha=0.25, backtrack_beta=0.5, max_backtrack_iters=40):
	''' Laplace approximation to sigmoid coefficient posteriors $phi$.
	'''

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
	return -jnp.sum(jnp.nan_to_num(y * jnp.log(lam) + (1 - y) * jnp.log(1 - lam))) \
	- jnp.sum(jnp.log(phi))/t + 1/2 * (phi - phi_prior) @ prec @ (phi - phi_prior)


