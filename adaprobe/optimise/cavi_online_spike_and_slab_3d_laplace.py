import numpy as np
from numba import njit, vectorize, float64
from .utils import sigmoid, get_mask, soften
from scipy.special import ndtr, ndtri
EPS = 1e-15

@njit
def cavi_online_spike_and_slab_3d_laplace(yk, stim, mu_prev, beta_prev, alpha_prev, shape_prev, rate_prev, phi_map_prev, phi_cov_prev, omega, C,
	interp=0.5, iters=10, verbose=True, newton_steps=5, seed=None, lam_update='monte-carlo', num_mc_samples=5, init_t=1e2, t_mult=1e1, 
	t_loops=10, focus_radius=30):
	"""Online-mode coordinate ascent variational inference for the adaprobe model.

	"""
	Lk, Ik = stim
	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prev.shape[0]

	mu = mu_prev.copy()
	beta = beta_prev.copy()
	alpha = alpha_prev.copy()
	shape = shape_prev
	rate = rate_prev
	phi_map = phi_map_prev.copy()
	phi_cov = phi_cov_prev.copy()

	mask = get_mask(N)
	focus = np.where(np.sqrt(np.sum(np.square(C - Lk), 1)) < focus_radius)[0]
	lamk = np.zeros(N)

	# Iterate CAVI updates
	for it in range(iters):
		beta[focus] = update_beta(alpha, lamk, shape, rate, beta_prev, focus)
		mu = update_mu(yk, mu, beta, alpha, lamk, shape, rate, mu_prev, beta_prev, mask, focus)
		alpha = update_alpha(yk, mu, beta, alpha, lamk, shape, rate, alpha_prev, mask, focus)
		lamk = update_lamk_monte_carlo(yk, mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov, mask, 
			omega, Lk, Ik, C, focus, num_mc_samples=num_mc_samples)
		shape, rate = update_sigma(yk, mu, beta, alpha, lamk, shape_prev, rate_prev, focus)
		phi_map, phi_cov = update_phi(lamk, phi_map_prev, phi_cov_prev, omega, Ik, Lk, C, focus, 
			newton_steps=newton_steps, init_t=init_t, t_mult=t_mult, t_loops=t_loops)

	mu = interp * mu + (1 - interp) * mu_prev
	beta = interp * beta + (1 - interp) * beta_prev
	alpha = interp * alpha + (1 - interp) * alpha_prev
	shape = interp * shape + (1 - interp) * shape_prev
	rate = interp * rate + (1 - interp) * rate_prev
	phi_map = interp * phi_map + (1 - interp) * phi_map_prev
	phi_cov = interp * phi_cov + (1 - interp) * phi_cov_prev

	return mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov

@njit
def update_beta(alpha, lamk, shape, rate, beta_prev, focus):
	return 1/np.sqrt(shape/rate * alpha[focus] * lamk[focus] + 1/(beta_prev[focus]**2))

@njit
def update_mu(yk, mu, beta, alpha, lamk, shape, rate, mu_prev, beta_prev, mask, focus):
	# N = mu.shape[0]
	sig = shape/rate
	for n in focus:
		mu[n] = (beta[n]**2) * (sig * alpha[n] * yk * lamk[n] - sig * alpha[n] * lamk[n] * np.sum(mu[mask[n]] * alpha[mask[n]] * lamk[mask[n]]) + mu_prev[n]/(beta_prev[n]**2))
	return mu

@njit
def update_alpha(yk, mu, beta, alpha, lamk, shape, rate, alpha_prev, mask, focus):
	# N = mu.shape[0]
	for n in focus:
		arg = -2 * mu[n] * yk * lamk[n] + 2 * mu[n] * lamk[n] * np.sum(mu[mask[n]] * alpha[mask[n]] * lamk[mask[n]]) + (mu[n]**2 + beta[n]**2) * lamk[n]
		alpha[n] = sigmoid(np.log((alpha_prev[n] + EPS)/(1 - alpha_prev[n] + EPS)) - shape/(2 * rate) * arg)
	return soften(alpha)

@njit
def update_lamk(yk, mu, beta, alpha, lamk, shape, rate, fk, mask, focus):
	# N = mu.shape[0]
	for n in range(focus):
		arg = -2 * yk * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * np.sum(mu[mask[n]] * alpha[mask[n]] * lamk[mask[n]]) + (mu[n]**2 + beta[n]**2) * alpha[n]
		lamk[n] = sigmoid(np.log((fk[n] + EPS)/(1 - fk[n] + EPS)) - shape/(2 * rate) * arg)
	return lamk

@vectorize([float64(float64, float64, float64)], nopython=True)
def _sample_truncated_normal(u, mean, sdev):
	"""Vectorised, JIT-compiled truncated normal samples using scipy's ndtr and ndtri
	"""
	return ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

@njit
def _sample_phi_independent_truncated_normals(phi_mapn, phi_covn, num_mc_samples=5):
	"""Returns (num_mc_samples x 2) sample of phi. Values are sampled from independent univariate truncated normals, due to
	intractability of sampling from truncated multivariate normals.
	"""
	samps = np.zeros((num_mc_samples, 2))
	for i in range(2):
		samps[:, i] = _sample_truncated_normal(np.random.rand(num_mc_samples), phi_mapn[i], np.sqrt(phi_covn[i, i]))
	return samps

@njit
def update_lamk_monte_carlo(yk, mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov, mask, omega, Lk, Ik, C, focus, num_mc_samples=5):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	N = lamk.shape[0]
	for n in focus:
		arg = -2 * yk * mu[n] * alpha[n] + 2 * mu[n] * alpha[n] * np.sum(mu[mask[n]] * alpha[mask[n]] * lamk[mask[n]]) + (mu[n]**2 + beta[n]**2) * alpha[n]
		mc_samps = _sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=num_mc_samples) # samples of phi for neuron n
		mcE = 0 # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			mk = Ik * np.exp(-np.sum(np.abs(Lk - C[n])/omega[n]))
			fn = sigmoid(mc_samps[indx, 0] * mk - mc_samps[indx, 1])
			fn = soften(fn)
			mcE = mcE + np.log(fn/(1 - fn))
		mcE = mcE/num_mc_samples
		lamk[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lamk	

@njit
def update_sigma(yk, mu, beta, alpha, lamk, prev_shape, prev_rate, focus):
	shape = prev_shape + 1/2
	rate = prev_rate + 1/2 * (np.square(yk - np.sum(mu * alpha * lamk)) \
		- np.sum(np.square(mu * alpha * lamk)) + np.sum((mu**2 + beta**2) * lamk * alpha))
	return shape, rate

@njit
def update_phi(lamk, phi_prior, phi_cov_prior, omega, Ik, Lk, C, focus, newton_steps=10, tol=1e-8, init_t=1e2, t_mult=1e1, t_loops=10, verbose=False):
	"""Returns updated sigmoid coefficients estimated using a sequential log-barrier penalty with backtracking Newton's method
	"""
	N = phi_prior.shape[0]
	phi_cov = phi_cov_prior.copy()
	phi = phi_prior.copy()
	for n in focus:
		# Reset t
		t = init_t
		for j in range(t_loops):
			for st in range(newton_steps):
				# Solve barrier problem with current t
				phi_new, phi_cov_new = _backtracking_newton_step_with_barrier(phi[n], lamk[n], t, phi_prior[n], phi_cov_prior[n], omega[n], Ik, Lk, C[n], verbose=verbose)
				if np.mean(np.abs(phi_new - phi[n])) < tol:
					# Newton's method converged
					break
				else:
					phi[n], phi_cov[n] = phi_new, phi_cov_new
			# sharpen log-barrier
			t = t * t_mult
	return phi, phi_cov

@njit
def _backtracking_newton_step_with_barrier(phi, lamk, t, phi_prior, phi_cov_prior, omega, Ik, Lk, C, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	"""Newton's method with backtracking line search. For fixed neuron n.
	"""
	mk = Ik * np.exp(-np.sum(np.abs(Lk - C)/omega))
	H_inv = np.zeros((2, 2))
	fk = sigmoid(phi[0] * mk - phi[1])
	phi_cov_prior_inv = np.linalg.inv(phi_cov_prior)

	# grad of negative log-likelihood
	j1 = -mk * (lamk - fk)
	j2 = lamk - fk
	J = np.array([j1, j2]) + phi_cov_prior_inv @ (phi - phi_prior) - 1/(t * phi)
	
	# hessian of negative log-likelihood
	h11 = mk**2 * fk * (1 - fk)
	h12 = -mk * fk * (1 - fk)
	h21 = h12
	h22 = fk * (1 - fk)
	H = np.array([[h11, h12], [h21, h22]]) + phi_cov_prior_inv + np.diag(1/(t * phi**2))
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	phi_cov_prior_inv_det = np.linalg.det(phi_cov_prior_inv)

	# begin backtracking
	step = 1
	for it in range(max_backtrack_iters):
		lhs = _negloglik_with_barrier(phi + step * v, t, lamk, mk, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det)
		rhs = _negloglik_with_barrier(phi, t, lamk, mk, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det) \
		+ backtrack_alpha * step * J.T @ v
		if np.isnan(lhs) or lhs > rhs:
			# shrink stepsize
			if verbose: print('shrinking step')
			step = backtrack_beta * step
		else:
			# proceed with Newton step
			if verbose: print('step size found', step)
			break
		if verbose and it == max_backtrack_iters - 1:
			print('no step size found')

	return phi + step * v, H_inv

@njit
def _negloglik_with_barrier(phi, t, lamk, mk, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det):
	"""Negative log-likelihood, for use with Newton's method. For fixed neuron n.
	"""
	fk = sigmoid(phi[0] * mk - phi[1])
	fk = soften(fk)
	nll = -lamk * np.log(fk) - (1 - lamk) * np.log(1 - fk) \
	+ 1/2 * (phi - phi_prior) @ phi_cov_prior_inv @ (phi - phi_prior).T + np.log(2 * np.pi) \
	- 1/2 * np.log(phi_cov_prior_inv_det) - np.sum(np.log(phi))/t
	return nll