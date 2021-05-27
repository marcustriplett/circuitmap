import numpy as np
from numba import njit, vectorize, float64
from .utils import sigmoid, get_mask, soften
from scipy.special import ndtr, ndtri, psi
EPS = 1e-15

# @njit
def cavi_online_spike_and_slab_joint_vi(yk, stim, mu_prev, Lam_prev, alpha_prev, shape_prev, rate_prev, omega, C,
	interp=0.5, iters=10, verbose=False, newton_steps=5, seed=None, lam_update='monte-carlo', num_mc_samples=5, init_t=1e2, t_mult=1e1, 
	t_loops=10, focus_radius=30):
	"""Online-mode coordinate-ascen variational inference for the adaprobe model.

	"""
	Lk, Ik = stim
	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prev.shape[0]

	mu = mu_prev.copy()
	Lam = Lam_prev.copy()
	alpha = alpha_prev.copy()
	shape = shape_prev
	rate = rate_prev

	mask = get_mask(N)
	focus = np.where(np.sqrt(np.sum(np.square(C - Lk), 1)) < focus_radius)[0]
	lamk = np.zeros(N)
	lamk[focus] = np.random.rand(focus.shape[0])
	# print('focus: ', focus + 1)

	# Iterate CAVI updates
	for it in range(iters):
		print(it)
		alpha = update_alpha(mu, Lam, alpha, lamk, shape, rate, mu_prev, Lam_prev, alpha_prev, mask, focus)
		lamk = update_lamk_monte_carlo(yk, mu, Lam, alpha, lamk, shape, rate, mask, omega, Lk, Ik, C, focus,
			num_mc_samples=num_mc_samples)
		mu, Lam = update_w_phi_joint(yk, alpha, mu_prev, Lam_prev, lamk, shape, rate, omega, Ik, Lk, C, mask, focus, 
			verbose=verbose)
		print('shape, rate before', shape, rate)
		# shape, rate = update_sigma(yk, mu, Lam, alpha, lamk, shape_prev, rate_prev, focus)
		print('shape, rate after', shape, rate)

	alpha = interp * alpha + (1 - interp) * alpha_prev
	mu = interp * mu + (1 - interp) * mu_prev
	Lam = interp * Lam + (1 - interp) * Lam_prev
	shape = interp * shape + (1 - interp) * shape_prev
	rate = interp * rate + (1 - interp) * rate_prev

	return mu, Lam, alpha, lamk, shape, rate

# @njit
def update_alpha(mu, Lam, alpha, lamk, shape, rate, mu_prev, Lam_prev, alpha_prev, mask, focus):
	"""
	FILL THIS IN
	"""
	logdet_Lam = np.log(np.linalg.det(Lam)) # fast determinants due to small covariance matrix dimensions
	logdet_Lam_prev = np.log(np.linalg.det(Lam_prev))
	Lam_prev_inv = np.linalg.inv(Lam_prev)
	for n in focus:
		arg = 1/2 * logdet_Lam[n] - 1/2 * logdet_Lam_prev[n] - 1/2 * (mu[n] - mu_prev[n]).T @ Lam_prev_inv[n] @ (mu[n] - mu_prev[n])\
			+ 3/2 - 1/2 * np.trace(Lam_prev_inv[n] @ Lam[n]) 
		alpha[n] = sigmoid(np.log((alpha_prev[n] + EPS)/(1 - alpha_prev[n] + EPS)) - shape/(2 * rate) * arg)

	return soften(alpha)	

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
def update_lamk_monte_carlo(yk, mu, Lam, alpha, lamk, shape, rate, mask, omega, Lk, Ik, C, focus, num_mc_samples=5):
	"""Infer latent spike rates using Monte Carlo samples of the sigmoid coefficients.
	"""
	for n in focus:
		arg = -2 * yk * mu[n, -1] * alpha[n] + 2 * mu[n, -1] * alpha[n] * np.sum(mu[mask[n], -1] * alpha[mask[n]] * lamk[mask[n]]) + (mu[n, -1]**2 + Lam[n][-1, -1]) * alpha[n]
		mc_samps = _sample_phi_independent_truncated_normals(mu[n, :-1], Lam[n][:-1, :-1], num_mc_samples=num_mc_samples) # samples of phi for neuron n
		mcE = 0 # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			fn = sigmoid(mc_samps[indx, 0] * Ik * np.exp(-omega[n] * np.sum(np.square(Lk - C[n]))) - mc_samps[indx, 1])
			fn = soften(fn)
			mcE = mcE + np.log(fn/(1 - fn))
		mcE = mcE/num_mc_samples
		lamk[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lamk

@njit
def update_sigma(yk, mu, Lam, alpha, lamk, prev_shape, prev_rate, focus):
	wvar = np.array([Lam_n[-1, -1] for Lam_n in Lam])
	shape = prev_shape + 1/2
	rate = prev_rate + 1/2 * (np.square(yk - np.sum(mu[:, -1] * alpha * lamk)) \
		- np.sum(np.square(mu[:, -1] * alpha * lamk)) + np.sum((mu[:, -1]**2 + wvar) * lamk * alpha))
	return shape, rate

@njit
def update_w_phi_joint(yk, alpha, mu_prior, Lam_prior, lamk, shape, rate, omega, Ik, Lk, C, mask, focus, newton_steps=10, tol=1e-8, init_t=1e2, t_mult=1e1, t_loops=10, verbose=False):
	"""Returns updated sigmoid coefficients estimated using a sequential log-barrier penalty with backtracking Newton's method
	"""
	N = mu_prior.shape[0]
	mu = mu_prior.copy()
	Lam = Lam_prior.copy()
	for n in focus:
		# Reset t
		t = init_t
		for j in range(t_loops):
			for st in range(newton_steps):
				# Solve barrier problem with current t
				mu_new, Lam_new = _backtracking_newton_step_with_barrier(n, t, mu, yk, alpha, lamk, shape, rate, mu_prior, Lam_prior, omega, Ik, Lk, C, mask, verbose=verbose)
				if np.mean(np.abs(mu_new - mu[n])) < tol:
					# Newton's method converged
					break
				else:
					mu[n], Lam[n] = mu_new, Lam_new
			# sharpen log-barrier
			t = t * t_mult
	return mu, Lam

@njit
def update_phi(lamk, phi_prior, phi_cov_prior, omega, Ik, Lk, C, focus, mask, newton_steps=10, tol=1e-8, init_t=1e2, t_mult=1e1, t_loops=10, verbose=False):
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
				phi_new, phi_cov_new = _backtracking_newton_step_with_barrier(phi[n], lamk, t, phi_prior[n], phi_cov_prior[n], omega[n], Ik, Lk, C[n], mask, verbose=verbose)
				if np.mean(np.abs(phi_new - phi[n])) < tol:
					# Newton's method converged
					break
				else:
					phi[n], phi_cov[n] = phi_new, phi_cov_new
			# sharpen log-barrier
			t = t * t_mult
	return phi, phi_cov

@njit
def _backtracking_newton_step_with_barrier(n, t, mu, yk, alpha, lamk, shape, rate, mu_prior, mu_cov_prior, omega, Ik, Lk, C, mask, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	""" Newton's method with backtracking line search. For fixed neuron n.

		Convention: mu[n, 0] = phi_n^0, mu[n, 1] = phi_n^1, mu[n, 2] = w_n

	"""
	mk = Ik * np.exp(-omega[n] * np.sum(np.square(C[n] - Lk)))
	fk = sigmoid(mu[n, 0] * mk - mu[n, 1])
	mu_cov_prior_inv_n = np.linalg.inv(mu_cov_prior[n])

	# grad of negative log-likelihood
	j1 = -mk * (lamk[n] - fk)
	j2 = lamk[n] - fk
	j3 = shape/rate * lamk[n] * (yk - mu[n, -1] - np.sum(mu[mask[n], -1] * lamk[mask[n]]))
	j_barrier = np.zeros(mu.shape[1])
	j_barrier[:-1] = 1/mu[n, :-1] # can adjust this to enforce log barrier on w too
	J = np.array([j1, j2, j3]) + mu_cov_prior_inv_n @ (mu[n] - mu_prior[n]) - 1/t * j_barrier
	
	# hessian of negative log-likelihood
	h11 = mk**2 * fk * (1 - fk)
	h12 = -mk * fk * (1 - fk)
	h21 = h12
	h22 = fk * (1 - fk)
	h33 = -shape/rate * lamk[n]
	h_barrier = np.zeros(mu.shape[1])
	h_barrier[:-1] = 1/mu[n, :-1]**2  # can adjust this to enforce log barrier on w too
	H = np.array([[h11, h12, 0], [h21, h22, 0], [0, 0, h33]]) + mu_cov_prior_inv_n + 1/t * np.diag(h_barrier)
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	# mu_cov_prior_inv_det = np.linalg.det(mu_cov_prior_inv_n)

	# begin backtracking
	step = 1
	next_mu = mu.copy()
	for it in range(max_backtrack_iters):
		next_mu[n] = mu[n] + step * v
		lhs = _negloglik_with_barrier(next_mu, n, t, yk, alpha, lamk, mk, shape, rate, mu_prior, mu_cov_prior_inv_n, mask)
		rhs = _negloglik_with_barrier(mu, n, t, yk, alpha, lamk, mk, shape, rate, mu_prior, mu_cov_prior_inv_n, mask) + backtrack_alpha * step * J.T @ v
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

	return next_mu[n], H_inv

@njit
def _negloglik_with_barrier(mu, n, t, yk, alpha, lamk, mk, shape, rate, mu_prior, mu_cov_prior_inv_n, mask):
	"""Negative log-likelihood up to an additive constant, for use with Newton's method. For fixed neuron n.
	"""
	fk = sigmoid(mu[n, 0] * mk - mu[n, 1])
	fk = soften(fk)
	nll = shape/(2 * rate) * mu[n, -1] * lamk[n] * (-2 * yk + mu[n, -1] + 2 * np.sum(alpha[mask[n]] * mu[mask[n], -1] * lamk[mask[n]])) \
	- lamk[n] * np.log(fk) - (1 - lamk[n]) * np.log(1 - fk) \
	+ 1/2 * (mu[n] - mu_prior[n]) @ mu_cov_prior_inv_n @ (mu[n] - mu_prior[n]).T \
	- np.sum(np.log(mu[n, :-1]))/t # exclude w from barrier
	return nll