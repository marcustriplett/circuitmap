import numpy as np
from numba import njit
from .utils import sigmoid, get_mask
EPS = 1e-13

# # @njit
# def cavi_online(y, stim, mu_prev, beta_prev, lam_prev, shape_prev, rate_prev, phi_prior, phi_cov_prior, omega, I, L, C, 
# 	iters=10, verbose=False, newton_steps=10, seed=None, lam_update='MAP', num_mc_samples=5):
# 	"""Online-mode coordinate-ascent variational inference for the adaprobe model.

# 	"""
# 	Lk, Ik = stim
# 	if seed is not None:
# 		np.random.seed(seed)

# 	# Initialise new params
# 	N = mu_prev.shape[0]
# 	K = len(I)
# 	mu = np.random.normal(0, 1, N)
# 	beta = np.exp(np.random.normal(0, 1, N))
# 	lam = np.random.rand(N, K)
# 	lam[:, :-1] = lam_prev
# 	shape = np.random.rand()
# 	rate = 5 + np.random.rand()
# 	phi_map = phi_prior.copy()
# 	phi_cov = phi_cov_prior.copy()
# 	mask = get_mask(N)

# 	# Iterate CAVI updates
# 	for it in range(iters):
# 		beta = update_beta(lam[:, -1], shape, rate, beta_prev)
# 		mu = update_mu(y[-1], mu, beta, lam[:, -1], shape, rate, mu_prev, beta_prev, mask)
# 		# maybe periodically run backwards updates? run backwards update if predictive performance is dropping
# 		if lam_update == 'MAP':
# 			# fk = sigmoid(phi_map[:, 0] * Ik * np.exp(-omega * np.sum(np.square(Lk - C), 1)) - phi_map[:, 1])
# 			# lam[:, -1] = update_lam_MAP_backward(y[-1], mu, beta, lam[:, -1], shape, rate, fk, mask)
# 			lam = update_lam_MAP_backward(y, mu, beta, lam, shape, rate, phi_map, omega, I, L, C, mask)
# 		elif lam_update == 'monte-carlo':
# 			lam = update_lam_monte_carlo_backward(y, mu, beta, lam, shape, rate, phi_map, phi_cov, mask, 
# 				omega, L, I, C, num_mc_samples=num_mc_samples)
# 		shape, rate = update_sigma(y[-1], mu, beta, lam[:, -1], shape_prev, rate_prev)
# 		phi_map, phi_cov = update_phi_accum(lam, phi_prior, phi_cov_prior, omega, I, L, C, 
# 			n_steps=newton_steps, verbose=verbose)

# 	return mu, beta, lam, shape, rate, phi_map, phi_cov

@njit
def cavi_online(yk, stim, mu_prev, beta_prev, shape_prev, rate_prev, phi_map_prev, phi_cov_prev, omega, C, 
	iters=10, verbose=False, newton_steps=10, seed=None, lam_update='MAP', num_mc_samples=5):
	"""Online-mode coordinate-ascent variational inference for the adaprobe model.

	"""
	Lk, Ik = stim
	if seed is not None:
		np.random.seed(seed)

	# Initialise new params
	N = mu_prev.shape[0]
	mu = np.random.normal(0, 1, N)
	beta = np.exp(np.random.normal(0, 1, N))
	lamk = np.random.rand(N)
	shape = np.random.rand()
	rate = 5 + np.random.rand()
	phi_map = phi_map_prev.copy()
	phi_cov = phi_cov_prev.copy()
	mask = get_mask(N)

	# Iterate CAVI updates
	for it in range(iters):
		beta = update_beta(lamk, shape, rate, beta_prev)
		mu = update_mu(yk, mu, beta, lamk, shape, rate, mu_prev, beta_prev, mask)
		if lam_update == 'MAP':
			fk = sigmoid(phi_map[:, 0] * Ik * np.exp(-omega * np.sum(np.square(Lk - C), 1)) - phi_map[:, 1])
			lamk = update_lamk_MAP(yk, mu, beta, lamk, shape, rate, fk, mask)
		elif lam_update == 'monte-carlo':
			lamk = update_lamk_monte_carlo(yk, mu, beta, lamk, shape, rate, phi_map, phi_cov, mask, \
				omega, Lk, Ik, C, num_mc_samples=num_mc_samples)
		shape, rate = update_sigma(yk, mu, beta, lamk, shape_prev, rate_prev)
		phi_map, phi_cov = update_phi(lamk, phi_map_prev, phi_cov_prev, omega, Ik, Lk, C, n_steps=newton_steps, verbose=verbose)

	return mu, beta, lamk, shape, rate, phi_map, phi_cov

@njit
def update_beta(lamk, shape, rate, beta_prev):
	return 1/np.sqrt(shape/rate * lamk + 1/(beta_prev**2))

@njit
def update_mu(yk, mu, beta, lamk, shape, rate, mu_prev, beta_prev, mask):
	N = mu.shape[0]
	sig = shape/rate
	for n in range(N):
		mu[n] = (beta[n]**2) * (sig * yk * lamk[n] - sig * lamk[n] * np.sum(mu[mask[n]] * lamk[mask[n]]) + mu_prev[n]/(beta_prev[n]**2))
	return mu

@njit
def update_lamk_MAP(yk, mu, beta, lamk, shape, rate, fk, mask):
	N = mu.shape[0]
	for n in range(N):
		arg = -2 * yk * mu[n] + 2 * mu[n] * np.sum(mu[mask[n]] * lamk[mask[n]]) + (mu[n]**2 + beta[n]**2)
		lamk[n] = sigmoid(np.log((fk[n] + EPS)/(1 - fk[n] + EPS)) - shape/(2 * rate) * arg)
	return lamk

@njit
def update_lam_MAP_backward(y, mu, beta, lam, shape, rate, phi_map, omega, I, L, C, mask):
	N, K = lam.shape
	f = np.zeros((N, K))
	for k in range(K):
		f[:, k] = sigmoid(phi_map[:, 0] * I[k] * np.exp(-omega * np.sum(np.square(L[k] - C), 1)) - phi_map[:, 1])
	for n in range(N):
		arg = -2 * y * mu[n] + 2 * mu[n] * np.sum(np.expand_dims(mu[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		lam[n] = sigmoid(np.log((f[n] + EPS)/(1 - f[n] + EPS)) - shape/(2 * rate) * arg)
	return lam

@njit
def _sample_phi(phi_mapn, phi_covn, num_mc_samples=1):
	"""Returns (num_mc_samples x 2) sample of phi.
	"""
	samps = np.zeros((num_mc_samples, 2))
	chol = np.linalg.cholesky(phi_covn)
	for n in range(num_mc_samples):
		samps[n] = chol @ np.random.standard_normal(phi_mapn.shape[0]) + phi_mapn
	return samps

@njit
def update_lamk_monte_carlo(yk, mu, beta, lamk, shape, rate, phi_map, phi_cov, mask, omega, Lk, Ik, C, num_mc_samples=5):
	N = mu.shape[0]
	for n in range(N):
		arg = -2 * yk * mu[n] + 2 * mu[n] * np.sum(mu[mask[n]] * lamk[mask[n]]) + (mu[n]**2 + beta[n]**2)
		mc_samps = _sample_phi(phi_map[n], phi_cov[n], num_mc_samples=num_mc_samples) # monte carlo samples of phi for neuron n
		mcE = 0 # monte carlo approximation of expectation
		for indx in range(num_mc_samples):
			fk = sigmoid(mc_samps[indx, 0] * Ik * np.exp(-omega[n] * np.sum(np.square(Lk - C[n]))) - mc_samps[indx, 1])
			mcE += np.log((fk + EPS)/(1 - fk + EPS))
		mcE /= num_mc_samples
		lamk[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lamk

@njit
def update_lam_monte_carlo_backward(y, mu, beta, lam, shape, rate, phi_map, phi_cov, mask, omega, L, I, C, num_mc_samples=5):
	N, K = lam.shape
	for n in range(N):
		arg = -2 * y * mu[n] + 2 * mu[n] * np.sum(np.expand_dims(mu[mask[n]], 1) * lam[mask[n]], 0) + (mu[n]**2 + beta[n]**2)
		mc_samps = _sample_phi(phi_map[n], phi_cov[n], num_mc_samples=num_mc_samples) # monte carlo samples of phi for neuron n
		mcE = np.zeros(K)
		for indx in range(num_mc_samples):
			fn = sigmoid(mc_samps[indx, 0] * I * np.exp(-omega[n] * np.sum(np.square(L - C[n]), 1)) - mc_samps[indx, 1])
			mcE = mcE + np.log((fn + EPS)/(1 - fn + EPS))
		mcE = mcE/num_mc_samples
		lam[n] = sigmoid(mcE - shape/(2 * rate) * arg)
	return lam

@njit
def update_sigma(yk, mu, beta, lamk, prev_shape, prev_rate):
	shape = prev_shape + 1/2
	rate = prev_rate + 1/2 * (np.square((yk - np.sum(mu * lamk))) \
		- np.sum(np.square(mu * lamk)) + np.sum((mu**2 + beta**2) * lamk))
	return shape, rate

@njit
def update_phi(lamk, prev_phi, prev_Lam, omega, Ik, Lk, C, n_steps=10, tol=1e-5, verbose=False):
	N = prev_phi.shape[0]
	Lam = np.zeros_like(prev_Lam)
	phi = prev_phi.copy()
	for n in range(N):
		for st in range(n_steps):
			phi[n], Lam[n] = _backtracking_newton_step_phi(phi[n], lamk[n], prev_phi[n], prev_Lam[n], omega[n], Ik, Lk, C[n], verbose=verbose)
	return phi, Lam

@njit
def update_phi_accum(lam, phi_prior, phi_cov_prior, omega, I, L, C, n_steps=10, tol=1e-5, verbose=False):
	N = phi_prior.shape[0]
	phi_cov = np.zeros_like(phi_cov_prior)
	phi = phi_prior.copy()
	for n in range(N):
		for st in range(n_steps):
			phi[n], phi_cov[n] = _backtracking_newton_step_phi_accum(phi[n], lam[n], phi_prior[n], phi_cov_prior[n], omega[n], I, L, C[n], verbose=verbose)
	return phi, phi_cov

@njit
def _newton_step_phi(phi, lamk, prev_phi, prev_Lam, omega, Ik, Lk, C, N, step=1e-1):
	"""Newton's method with fixed step size.
	"""
	mk = Ik * np.exp(-omega * np.sum(np.square(C - Lk), 1))
	H_inv = np.zeros((N, 2, 2))
	for n in range(N):
		fk = sigmoid(phi[:, 0] * mk - phi[:, 1])
		prev_Lam_inv = np.linalg.inv(prev_Lam[n])

		# grad of negative log-likelihood
		j1 = -mk[n] * (lamk[n] - fk[n])
		j2 = (lamk[n] - fk[n])
		J = np.array([j1, j2]) + prev_Lam_inv @ (phi[n] - prev_phi[n])
		
		# hessian of negative log-likelihood
		h11 = mk[n]**2 * fk[n] * (1 - fk[n])
		h12 = -mk[n] * fk[n] * (1 - fk[n])
		h21 = h12
		h22 = fk[n] * (1 - fk[n])
		H = np.array([[h11, h12], [h21, h22]]) + prev_Lam_inv
		
		H_inv[n] = np.linalg.inv(H)
		v = -H_inv[n] @ J # Newton direction

		phi[n] = phi[n] + step * v

	return phi, H_inv

@njit
def _backtracking_newton_step_phi(phi, lamk, prev_phi, prev_Lam, omega, Ik, Lk, C, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	"""Newton's method with backtracking line search. One neuron, one trial.
	"""
	mk = Ik * np.exp(-omega * np.sum(np.square(C - Lk)))
	H_inv = np.zeros((2, 2))
	fk = sigmoid(phi[0] * mk - phi[1])
	prev_Lam_inv = np.linalg.inv(prev_Lam)

	# grad of negative log-likelihood
	j1 = -mk * (lamk - fk)
	j2 = (lamk - fk)
	J = np.array([j1, j2]) + prev_Lam_inv @ (phi - prev_phi)
	
	# hessian of negative log-likelihood
	h11 = mk**2 * fk * (1 - fk)
	h12 = -mk * fk * (1 - fk)
	h21 = h12
	h22 = fk * (1 - fk)
	H = np.array([[h11, h12], [h21, h22]]) + prev_Lam_inv
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	prev_Lam_inv_det = np.linalg.det(prev_Lam_inv)

	# begin backtracking
	step = 1
	for it in range(max_backtrack_iters):
		lhs = _negloglik(phi + step * v, lamk, mk, prev_phi, prev_Lam_inv, prev_Lam_inv_det)
		rhs = _negloglik(phi, lamk, mk, prev_phi, prev_Lam_inv, prev_Lam_inv_det) \
			+ backtrack_alpha * step * J.T @ v
		if lhs > rhs:
			# shrink stepsize
			if verbose: print('shrinking step')
			step = backtrack_beta * step
		else:
			# proceed with Newton step
			if verbose: print('step size found', step)
			break

	return phi + step * v, H_inv

@njit
def _backtracking_newton_step_phi_accum(phi, lam, phi_prior, phi_cov_prior, omega, I, L, C, backtrack_alpha=0.25, 
	backtrack_beta=0.5, max_backtrack_iters=15, verbose=False):
	"""Newton's method with backtracking line search for fixed neuron n.
	"""
	m = I * np.exp(-omega * np.sum(np.square(C - L), 1))
	H_inv = np.zeros((2, 2))
	f = sigmoid(phi[0] * m - phi[1])
	phi_cov_prior_inv = np.linalg.inv(phi_cov_prior)

	# grad of negative log-likelihood
	j1 = -np.sum(m * (lam - f))
	j2 = np.sum(lam - f)
	J = np.array([j1, j2]) + phi_cov_prior_inv @ (phi - phi_prior)
	
	# hessian of negative log-likelihood
	h11 = np.sum(m**2 * f * (1 - f))
	h12 = -np.sum(m * f * (1 - f))
	h21 = h12
	h22 = np.sum(f * (1 - f))
	H = np.array([[h11, h12], [h21, h22]]) + phi_cov_prior_inv
	
	H_inv = np.linalg.inv(H)
	v = -H_inv @ J # Newton direction
	phi_cov_prior_inv_det = np.linalg.det(phi_cov_prior_inv)

	# begin backtracking
	step = 1
	for it in range(max_backtrack_iters):
		lhs = _negloglik_accum(phi + step * v, lam, m, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det)
		rhs = _negloglik_accum(phi, lam, m, phi_prior, phi_cov_prior, phi_cov_prior_inv_det) \
			+ backtrack_alpha * step * J.T @ v
		if lhs > rhs:
			# shrink stepsize
			if verbose: print('shrinking step')
			step = backtrack_beta * step
		else:
			# proceed with Newton step
			if verbose: print('step size found', step)
			break

	return phi + step * v, H_inv

@njit
def _negloglik(phi, lamk, mk, prev_phi, prev_Lam_inv, prev_Lam_inv_det):
	"""Negative log-likelihood, for use with Newton's method.
	"""
	f = sigmoid(phi[0] * mk - phi[1])
	nll = -lamk * np.log(f + EPS) - (1 - lamk) * np.log(1 - f + EPS) \
	+ 1/2 * (phi - prev_phi) @ prev_Lam_inv @ (phi - prev_phi).T + np.log(2 * np.pi) \
	- 1/2 * np.log(prev_Lam_inv_det)
	return nll

@njit
def _negloglik_accum(phi, lam, m, phi_prior, phi_cov_prior_inv, phi_cov_prior_inv_det):
	"""Negative log-likelihood, for use with Newton's method.
	"""
	f = sigmoid(phi[0] * m - phi[1])
	nll = -np.sum(lam * np.log(f + EPS)) - np.sum((1 - lam) * np.log(1 - f + EPS)) \
	+ 1/2 * (phi - phi_prior) @ phi_cov_prior_inv @ (phi - phi_prior).T + np.log(2 * np.pi) \
	- 1/2 * np.log(phi_cov_prior_inv_det)
	return nll