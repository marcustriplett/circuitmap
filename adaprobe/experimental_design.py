import numpy as np
from numba import njit, float64, prange
from adaprobe.optimise.cavi_online_spike_and_slab import cavi_online_spike_and_slab
from adaprobe.optimise.cavi_online_spike_and_slab_3d_omega import cavi_online_spike_and_slab_3d_omega
from adaprobe.optimise.cavi_online_spike_and_slab import _sample_phi_independent_truncated_normals
from adaprobe.optimise.utils import sigmoid
import scipy.special; import numba_special # overload scipy special funcs
import ctypes
from numba.extending import get_cython_function_address as getaddr
from scipy.stats import norm
from functools import partial

psi_fnaddr = getaddr("scipy.special.cython_special", "__pyx_fuse_1psi")
psi_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
psi = psi_ftype(psi_fnaddr)

EPS = 1e-15

def model_evidence(y, model, I, L, num_mc_samples=5):
	N, K = model.n_presynaptic, len(I)
	m = np.array([I * np.exp(-model.state['omega'][n] * np.sum(np.square(L - model.cell_locs[n]), 1)) for n in range(N)])
	phi = np.array([
		_sample_phi_independent_truncated_normals(model.state['phi_map'][n], model.state['phi_cov'][n], num_mc_samples=num_mc_samples) for n in range(N)
	])
	f = np.zeros((num_mc_samples, N, K))
	s = np.zeros_like(f)
	w = np.zeros((num_mc_samples, N))
	sig = np.zeros(num_mc_samples)
	llh = np.zeros(num_mc_samples)
	for i in range(num_mc_samples):
		f[i] = sigmoid(np.array([phi[n, i, 0] * m[n] - phi[n, i, 1] for n in range(N)]))
		s[i] = np.random.rand(N, K) <= f[i]
		w[i] = (np.random.rand(N) <= model.state['alpha']) * np.random.normal(model.state['mu'], model.state['beta'])
		sig[i] = np.sqrt(1/np.random.gamma(model.state['shape'], 1/model.state['rate']))
		llh[i] = np.sum(norm.pdf(y, w[i] @ s[i], sig[i]))
	return np.mean(llh)

@njit
def _get_mk_3d_omega(Omega, C, N, Ik, Lk):
    mk = np.zeros(N)
    for n in prange(N):
        mk[n] = Ik * np.exp(-(Lk - C[n]) @ Omega[n] @ (Lk - C[n]))
    return mk

# @njit
def get_m_3d_omega(I, L, K, N, get_mk):
    m = np.zeros((N, K))
    for k in range(K):
        m[:, k] = get_mk(I[k], L[k])
    return m

def model_evidence_omega_3d(y, model, I, L, num_mc_samples=5):
	K = len(I)
	N = len(model.cell_locs)
	get_mk_3d_omega = partial(_get_mk_3d_omega, model.state['Omega'], model.cell_locs, N)
	m = get_m_3d_omega(I, L, K, N, get_mk_3d_omega)
	phi = np.array([_sample_phi_independent_truncated_normals(
			model.state['phi_map'][n], model.state['phi_cov'][n], num_mc_samples=num_mc_samples
		) for n in range(N)])
	f = np.zeros((num_mc_samples, N, K))
	s = np.zeros_like(f)
	w = np.zeros((num_mc_samples, N))
	sig = np.zeros(num_mc_samples)
	llh = np.zeros(num_mc_samples)
	for i in range(num_mc_samples):
		f[i] = sigmoid(np.array([phi[n, i, 0] * m[n] - phi[n, i, 1] for n in range(N)]))
		s[i] = np.random.rand(N, K) <= f[i]
		w[i] = (np.random.rand(N) <= model.state['alpha']) * np.random.normal(model.state['mu'], model.state['beta'])
		sig[i] = np.sqrt(1/np.random.gamma(model.state['shape'], 1/model.state['rate']))
		llh[i] = np.sum(norm.pdf(y, w[i] @ s[i], sig[i]))
	return np.mean(llh)

def model_evidence_3d_laplace(y, model, I, L, num_mc_samples=5):
    N = model.n_presynaptic
    K = len(I)
    m = np.zeros((N, K))
    for n in range(N):
        m[n] = I * np.exp(-np.sum(np.abs(L - model.cell_locs[n])/model.state['omega'][n], 1))
    phi = np.array([_sample_phi_independent_truncated_normals(
        model.state['phi_map'][n], model.state['phi_cov'][n], num_mc_samples=num_mc_samples
    ) for n in range(N)])
    f = np.zeros((num_mc_samples, N, K))
    s = np.zeros_like(f)
    w = np.zeros((num_mc_samples, N))
    sig = np.zeros(num_mc_samples)
    llh = np.zeros(num_mc_samples)
    for i in range(num_mc_samples):
        f[i] = sigmoid(np.array([phi[n, i, 0] * m[n] - phi[n, i, 1] for n in range(N)]))
        s[i] = np.random.rand(N, K) <= f[i]
        w[i] = (np.random.rand(N) <= model.state['alpha']) * np.random.normal(model.state['mu'], model.state['beta'])
        sig[i] = np.sqrt(1/np.random.gamma(model.state['shape'], 1/model.state['rate']))
        llh[i] = np.sum(norm.pdf(y, w[i] @ s[i], sig[i]))
    return np.mean(llh)

def optogenetic_receptive_field_on_plane(grid2d, plane, Ik, model):
	"""Optogenetic receptive field.
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	npts = grid.shape[0]
	N = len(model.cell_locs)
	cell_ids_on_plane = np.arange(len(model.cell_locs))[model.cell_locs[:, 2] == plane]
	spike_prob = np.zeros((N, npts))
	for n in cell_ids_on_plane:
		spike_prob[n] = model.state['alpha'][n] * sigmoid(model.state['phi_map'][n, 0] * Ik * np.exp(-model.state['omega'][n] * np.sum(np.square(grid - model.cell_locs[n]), 1)) - model.state['phi_map'][n, 1])
	
	return np.sum(spike_prob, 0)

def synaptic_weights_on_plane(grid2d, plane, Ik, model):
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	npts = grid.shape[0]
	N = len(model.cell_locs)
	cell_ids_on_plane = np.arange(len(model.cell_locs))[model.cell_locs[:, 2] == plane]
	resp = np.zeros((N, npts))
	for n in cell_ids_on_plane:
		resp[n] = model.state['mu'][n] * model.state['alpha'][n] * sigmoid(model.state['phi_map'][n, 0] * Ik * np.exp(-model.state['omega'][n] * np.sum(np.square(grid - model.cell_locs[n]), 1)) - model.state['phi_map'][n, 1])
	
	return np.sum(resp, 0)

def posterior_predictive_response(Lk, Ik, model, n_samples=10):
	"""
	"""
	return _posterior_predictive_response(Lk, Ik, model.n_presynaptic, model.cell_locs, model.state['mu'], model.state['beta'], model.state['alpha'], model.state['omega'], model.state['shape'], model.state['rate'],
		model.state['phi_map'], model.state['phi_cov'], n_samples=n_samples)

@njit(parallel=True)
def _posterior_predictive_response(Lk, Ik, N, cell_locs, mu, beta, alpha, omega, shape, rate, phi_map, phi_cov, n_samples=10):
	"""
	"""
	spike_prob = np.zeros(N)
	ysamp = np.zeros(n_samples)
	for indx in prange(n_samples):
		u = mu + np.random.standard_normal(N) * beta
		gam = (np.random.rand(N) <= alpha).astype(float64)
		phi = [_sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)[0] for n in range(N)]
		for n in range(N):
			spike_prob[n] = sigmoid(phi[n][0] * Ik * np.exp(-omega[n] * np.sum(np.square(Lk - cell_locs[n]))) - phi[n][1])
		s = (np.random.rand(N) <= spike_prob).astype(float64)
		sig = np.random.gamma(shape, 1/rate)
		ysamp[indx] = np.random.standard_normal() * np.sqrt(1/sig) + np.sum(u * gam * s)
	return np.mean(ysamp)

def posterior_predictive_map_on_grid(grid, Ik, model, n_samples=10):
	"""
	"""
	return np.mean(_posterior_predictive_map(grid, Ik, model.n_presynaptic, model.cell_locs, model.state['mu'], model.state['beta'], model.state['alpha'],
		 model.state['omega'], model.state['shape'], model.state['rate'], model.state['phi_map'], model.state['phi_cov'], n_samples=n_samples), 0)

def posterior_predictive_map_on_plane(grid2d, plane=0, power=70, model=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	return np.mean(_posterior_predictive_map(grid, power, model.n_presynaptic, model.cell_locs, model.state['mu'], model.state['beta'], model.state['alpha'],
		model.state['omega'], model.state['shape'], model.state['rate'], model.state['phi_map'], model.state['phi_cov'], n_samples=n_samples), 0)

def posterior_predictive_map_on_plane_3d_omega(grid2d, plane=0, power=70, model=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	return np.mean(_posterior_predictive_map_3d_omega(grid, power, model.n_presynaptic, model.cell_locs, model.state['mu'], model.state['beta'], model.state['alpha'],
		model.state['Omega'], model.state['shape'], model.state['rate'], model.state['phi_map'], model.state['phi_cov'], n_samples=n_samples), 0)

def posterior_predictive_map_on_plane_3d_laplace(grid2d, plane=0, power=70, model=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	return np.mean(_posterior_predictive_map_3d_laplace(grid, power, model.n_presynaptic, model.cell_locs, model.state['mu'], model.state['beta'], model.state['alpha'],
		model.state['omega'], model.state['shape'], model.state['rate'], model.state['phi_map'], model.state['phi_cov'], n_samples=n_samples), 0)

def posterior_predictive_map_on_plane_bayes_ensemble(grid2d, y, plane=0, power=70, I_hist=None, L_hist=None, models=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	ev = np.array([model_evidence(y, model, I_hist, L_hist, num_mc_samples=n_samples) for model in models])
	posterior_weights = ev/np.sum(ev)

	labels = ['mu', 'beta', 'alpha', 'omega', 'shape', 'rate', 'phi_map', 'phi_cov']
	params = [None] * len(labels)

	for i in range(len(labels)):
		params[i] = np.sum([models[m].state[labels[i]] * posterior_weights[m] for m in range(len(models))], 0)

	mu, beta, alpha, omega, shape, rate, phi_map, phi_cov = params
	
	return np.mean(_posterior_predictive_map(grid, power, models[0].n_presynaptic, models[0].cell_locs, mu, beta, alpha, omega, 
		shape, rate, phi_map, phi_cov, n_samples=n_samples), 0)

def posterior_predictive_map_on_plane_bayes_ensemble_3d_omega(grid2d, y, plane=0, power=70, I_hist=None, L_hist=None, models=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	ev = np.array([model_evidence_omega_3d(y, model, I_hist, L_hist, num_mc_samples=n_samples) for model in models])
	posterior_weights = ev/np.sum(ev)

	labels = ['mu', 'beta', 'alpha', 'Omega', 'shape', 'rate', 'phi_map', 'phi_cov']
	params = [None] * len(labels)

	for i in range(len(labels)):
		params[i] = np.sum([models[m].state[labels[i]] * posterior_weights[m] for m in range(len(models))], 0)

	mu, beta, alpha, Omega, shape, rate, phi_map, phi_cov = params
	
	return np.mean(_posterior_predictive_map_3d_omega(grid, power, models[0].n_presynaptic, models[0].cell_locs, mu, beta, alpha, Omega, 
		shape, rate, phi_map, phi_cov, n_samples=n_samples), 0)

def posterior_predictive_map_on_plane_bayes_ensemble_3d_laplace(grid2d, y, plane=0, power=70, I_hist=None, L_hist=None, models=None, n_samples=10):
	"""
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))]
	ev = np.array([model_evidence_3d_laplace(y, model, I_hist, L_hist, num_mc_samples=n_samples) for model in models])
	posterior_weights = ev/np.sum(ev)

	labels = ['mu', 'beta', 'alpha', 'omega', 'shape', 'rate', 'phi_map', 'phi_cov']
	params = [None] * len(labels)

	for i in range(len(labels)):
		params[i] = np.sum([models[m].state[labels[i]] * posterior_weights[m] for m in range(len(models))], 0)

	mu, beta, alpha, omega, shape, rate, phi_map, phi_cov = params
	
	return np.mean(_posterior_predictive_map_3d_laplace(grid, power, models[0].n_presynaptic, models[0].cell_locs, mu, beta, alpha, omega, 
		shape, rate, phi_map, phi_cov, n_samples=n_samples), 0)

def _posterior_predictive_map(grid, Ik, N, cell_locs, mu, beta, alpha, omega, shape, rate, phi_map, phi_cov, n_samples=10):
	"""Use samples from the posterior predictive distribution to build predictive map.
	"""
	npts = grid.shape[0]
	spike_prob = np.zeros((N, npts))
	ysamp = np.zeros((n_samples, npts))
	for indx in range(n_samples):
		u = np.random.standard_normal(N) * beta + mu
		gam = (np.random.rand(N) <= alpha).astype(float)
		phi = [_sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)[0] for n in range(N)]
		for n in range(N):
			spike_prob[n] = sigmoid(phi[n][0] * Ik * np.exp(-omega[n] * np.sum(np.square(grid - cell_locs[n]), 1)) - phi[n][1])
		s = (np.random.rand(N, npts) <= spike_prob).astype(float)
		sig = np.random.gamma(shape, 1/rate, npts)
		ysamp[indx] = np.random.standard_normal(npts) * np.sqrt(1/sig) + np.sum(np.expand_dims(u * gam, 1) * s, 0)
	return ysamp

@njit(parallel=False)
def _posterior_predictive_map_3d_omega(grid, Ik, N, cell_locs, mu, beta, alpha, Omega, shape, rate, phi_map, phi_cov, n_samples=10):
	"""Use samples from the posterior predictive distribution to build predictive map.
	"""
	npts = grid.shape[0]
	spike_prob = np.zeros((N, npts))
	ysamp = np.zeros((n_samples, npts))
	for indx in prange(n_samples):
		u = np.random.standard_normal(N) * beta + mu
		gam = (np.random.rand(N) <= alpha).astype(float64)
		phi = [_sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)[0] for n in range(N)]
		for n in range(N):
			spike_prob[n] = np.array([sigmoid(phi[n][0] * Ik * np.exp(-(g - cell_locs[n]) @ Omega[n]/Ik @ (g - cell_locs[n]).T) - phi[n][1]) for g in grid])
		s = (np.random.rand(N, npts) <= spike_prob).astype(float64)
		sig = np.random.gamma(shape, 1/rate, npts)
		ysamp[indx] = np.random.standard_normal(npts) * np.sqrt(1/sig) + np.sum(np.expand_dims(u * gam, 1) * s, 0)
	return ysamp

@njit(parallel=False)
def _posterior_predictive_map_3d_laplace(grid, Ik, N, cell_locs, mu, beta, alpha, omega, shape, rate, phi_map, phi_cov, n_samples=10):
	"""Use samples from the posterior predictive distribution to build predictive map.
	"""
	npts = grid.shape[0]
	spike_prob = np.zeros((N, npts))
	ysamp = np.zeros((n_samples, npts))
	for indx in prange(n_samples):
		u = np.random.standard_normal(N) * beta + mu
		gam = (np.random.rand(N) <= alpha).astype(float64)
		phi = [_sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)[0] for n in range(N)]
		for n in range(N):
			spike_prob[n] = sigmoid(phi[n][0] * Ik * np.exp(-np.sum(np.abs(grid - cell_locs[n])/omega[n], 1)) - phi[n][1])
		s = (np.random.rand(N, npts) <= spike_prob).astype(float64)
		sig = np.random.gamma(shape, 1/rate, npts)
		ysamp[indx] = np.random.standard_normal(npts) * np.sqrt(1/sig) + np.sum(np.expand_dims(u * gam, 1) * s, 0)
	return ysamp

def posterior_predictive_error(Lk, Ik, model, method='cavi_online_spike_and_slab', n_samples=10):
	"""Calculate expected posterior entropy reduction for the spike-and-slab adaprobe model.
	"""
	Hw = np.sum(np.log(beta) - alpha * np.log(alpha + EPS) - (1 - alpha) * np.log(1 - alpha + EPS))
	entrops = np.zeros(n_samples)
	for i in range(n_samples):
		yk = _get_sample_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik)
		out = cavi_online_spike_and_slab(
			yk, (Lk, Ik), mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C
		)
		entrops[i] = np.sum(np.log(next_beta) - next_alpha * np.log(next_alpha + EPS) \
			- (1 - next_alpha) * np.log(1 - next_alpha + EPS))
	return Hw - np.mean(entrops)

def expected_entropy_decrease(Lk, Ik, model, method='cavi_online', n_samples=10):
	"""Calculate the expected reduction in posterior entropy.
	"""
	mu = model.state['mu']
	beta = model.state['beta']
	shape = model.state['shape']
	rate = model.state['rate']
	phi_map = model.state['phi_map']
	phi_cov = model.state['phi_cov']
	omega = model.state['omega']
	C = model.cell_locs

	if method == 'cavi_online':
		return _expected_entropy_decrease(mu, beta, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik, n_samples)
	elif method == 'cavi_online_spike_and_slab':
		return _expected_entropy_decrease_spike_and_slab(mu, beta, model.state['alpha'], shape, rate, phi_map, 
			phi_cov, omega, C, Lk, Ik, n_samples)
	else:
		raise Exception("Kwarg 'method' must be either 'cavi_online' or 'cavi_online_spike_and_slab'.")

def expected_entropy_decrease_spike_and_slab_3d(grid, I, model, method='cavi_online_spike_and_slab_3d_omega', n_samples=20):
	"""Calculate the expected reduction in posterior entropy of synapses in three-dimensional utility space, averaging
	over other parameters.
	"""
	mu = model.state['mu']
	beta = model.state['beta']
	alpha = model.state['alpha']
	shape = model.state['shape']
	rate = model.state['rate']
	phi_map = model.state['phi_map']
	phi_cov = model.state['phi_cov']
	Omega = model.state['Omega']
	C = model.cell_locs

	return _expected_entropy_decrease_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, 
		grid, I, n_samples)

def expected_universal_entropy_decrease_spike_and_slab_3d(grid, I, model, method='cavi_online_spike_and_slab_3d_omega', n_samples=20):
	"""Calculate the expected reduction in posterior entropy of synapses in three-dimensional utility space, averaging
	over other parameters.
	"""
	mu = model.state['mu']
	beta = model.state['beta']
	alpha = model.state['alpha']
	shape = model.state['shape']
	rate = model.state['rate']
	phi_map = model.state['phi_map']
	phi_cov = model.state['phi_cov']
	Omega = model.state['Omega']
	C = model.cell_locs

	return _expected_universal_entropy_decrease_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, 
		grid, I, n_samples)

def expected_entropy_decrease_on_grid(grid, I, model, method='cavi_online_spike_and_slab', n_samples=20):
	"""Calculate the expected reduction in posterior entropy of synapses, averaging over other parameters.
	"""
	mu = model.state['mu']
	beta = model.state['beta']
	shape = model.state['shape']
	rate = model.state['rate']
	phi_map = model.state['phi_map']
	phi_cov = model.state['phi_cov']
	omega = model.state['omega']
	C = model.cell_locs

	if method == 'cavi_online':
		return _expected_entropy_decrease_on_grid(mu, beta, shape, rate, phi_map, phi_cov, omega, C, grid, I,
		n_samples)
	elif method == 'cavi_online_spike_and_slab':
		return _expected_entropy_decrease_spike_and_slab_on_grid(mu, beta, model.state['alpha'], shape, rate,
			phi_map, phi_cov, omega, C, grid, I, n_samples)
	else:
		raise Exception("Kwarg 'method' must be either 'cavi_online' or 'cavi_online_spike_and_slab'.")

def expected_universal_entropy_decrease_on_grid(grid, I, model, n_samples=20):
	"""Calculate expected reduction in posterior entropy of all parameters.
	"""
	return _expected_universal_entropy_decrease_on_grid(model.state['mu'], model.state['beta'], model.state['alpha'],
		model.state['shape'], model.state['rate'], model.state['phi_map'], model.state['phi_cov'], model.state['omega'],
		model.cell_locs, grid, I, n_samples)

def weighted_spike_count(Lk, Ik, model):
	"""Maximise number of neurons with probability of firing around 50%, weighted by posterior entropy of their connections.
	"""
	pred_lam = sigmoid(model.state['phi_map'][:, 0] * Ik * np.exp(model.state['omega'] - np.sum(np.square(Lk - model.cell_locs), 1)) - model.state['phi_map'][:, 1])
	Hw = np.log(model.state['beta']) - model.state['alpha'] * np.log(model.state['alpha'] + EPS) - (1 - model.state['alpha']) * np.log(1 - model.state['alpha'] + EPS)
	return np.sum(np.square(pred_lam - 1/2) * Hw)

def weighted_spike_count_on_grid(grid, Ik, model):
	"""Calculates entropy-weighted spike count utility over grid.
	"""
	return _weighted_spike_count_on_grid(grid, Ik, model.state['alpha'], model.state['beta'], model.state['phi_map'], model.state['omega'], model.cell_locs)

def entropy_weighted_expected_firing_rate(Lk, Ik, model):
	"""Fixed Lk, Ik
	"""
	alpha = model.state['alpha']
	Hw = 1/2 * np.log(2 * np.pi * np.exp(1) * model.state['beta']**2) - alpha * np.log(alpha + EPS) - (1 - alpha) * np.log(1 - alpha + EPS)
	pred_lam = sigmoid(model.state['phi_map'][:, 0] * Ik * np.exp(-model.state['omega'] * np.sum(np.square(Lk - model.cell_locs), 1)) - model.state['phi_map'][:, 1])
	normalised_spike_prob = (alpha * pred_lam) / np.sum(alpha * pred_lam)

	return np.sum(Hw * normalised_spike_prob * pred_lam)

def entropy_weighted_expected_firing_rate_on_grid(grid, Ik, model):
	"""
	"""
	return _entropy_weighted_expected_firing_rate_on_grid(grid, Ik, model.state['alpha'], model.state['beta'], model.state['phi_map'], model.state['omega'], model.cell_locs)

# @njit
def _entropy_weighted_expected_firing_rate_on_grid(grid, Ik, _alpha, beta, phi, omega, C):
	"""Approximation to the infomax utility.
	"""
	npts = grid.shape[0]
	N = C.shape[0]
	alpha = _alpha.copy()
	alpha[alpha == 0] = EPS
	alpha[alpha == 1] = 1 - EPS
	Hw = 1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) - alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)
	pred_lam = np.zeros((N, npts))
	normalised_spike_prob = np.zeros((N, npts))
	for n in range(N):
		pred_lam[n] = sigmoid(phi[n, 0] * Ik * np.exp(-omega[n] * np.sum(np.square(grid - C[n]), 1)) - phi[n, 1]) # expected spikes
		normalised_spike_prob[n] = (alpha[n] * pred_lam[n]) / np.sum(alpha[:, None] * pred_lam, 0)
	
	# return np.sum(Hw[:, None] * normalised_spike_prob, 0)
	return np.sum(Hw[:, None] * pred_lam * normalised_spike_prob, 0)

def entropy_weighted_expected_firing_rate_on_plane(grid, plane, Ik, model):
	"""
	"""
	return _entropy_weighted_expected_firing_rate_on_plane(grid, plane, Ik, model.state['alpha'], model.state['beta'], model.state['phi_map'], model.state['omega'], model.cell_locs)

def _entropy_weighted_expected_firing_rate_on_plane(grid2d, plane, Ik, _alpha, beta, phi, omega, C):
	"""Approximation to the infomax utility.
	"""
	grid = np.c_[grid2d, plane * np.ones(len(grid2d))] # recast grid in 3d
	npts = grid.shape[0]
	cell_locs_on_plane = C[C[:, 2] == plane][:, :2]
	N = cell_locs_on_plane.shape[0]
	cell_ids_on_plane = np.arange(len(C))[C[:, 2] == plane]
	alpha = _alpha.copy()
	alpha[alpha == 0] = EPS
	alpha[alpha == 1] = 1 - EPS
	Hw = 1/2 * np.log(2 * np.pi * np.exp(1) * beta[cell_ids_on_plane]**2) - alpha[cell_ids_on_plane] * np.log(alpha[cell_ids_on_plane]) \
		- (1 - alpha[cell_ids_on_plane]) * np.log(1 - alpha[cell_ids_on_plane])
	pred_lam = np.zeros((N, npts))
	normalised_spike_prob = np.zeros((N, npts))
	for _n in range(N):
		n = cell_ids_on_plane[_n]
		pred_lam[_n] = sigmoid(phi[n, 0] * Ik * np.exp(-omega[n] * np.sum(np.square(grid - C[n]), 1)) - phi[n, 1]) # expected spikes
		normalised_spike_prob[_n] = (alpha[n] * pred_lam[_n]) / np.sum(alpha[cell_ids_on_plane, None] * pred_lam, 0)
	
	return np.sum(Hw[:, None] * pred_lam * normalised_spike_prob, 0)

@njit
def _weighted_spike_count_on_grid(grid, Ik, alpha, beta, phi, omega, C):
	npts = grid.shape[0]
	N = C.shape[0]
	wsc_grid = np.zeros(npts)
	Hw = 1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) - alpha * np.log(alpha + EPS) - (1 - alpha) * np.log(1 - alpha + EPS)
	for n in range(N):
		pred_lam = sigmoid(phi[n, 0] * Ik * np.exp(-omega[n] * np.sum(np.square(grid - C[n]), 1)) - phi[n, 1])
		wsc_grid = wsc_grid + np.square(pred_lam - 1/2) / Hw[n]
	return wsc_grid

@njit
def _expected_entropy_decrease(mu, beta, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik, 
	n_samples):
	"""Calculate expected posterior entropy reduction for the adaprobe model.
	"""
	Hw = np.sum(np.log(beta))
	entrops = np.zeros(n_samples)
	for i in range(n_samples):
		y = _get_sample(mu, beta, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik)
		out = cavi_online(
			y, (Lk, Ik), mu, beta, shape, rate, phi_map, phi_cov, omega, C
		)
		next_beta = out[1]
		entrops[i] = np.sum(np.log(next_beta))
	return Hw - np.mean(entrops)

@njit
def _expected_entropy_decrease_on_grid(mu, beta, shape, rate, phi_map, phi_cov, omega, C, grid, I, 
	n_samples):
	"""Calculate expected posterior entropy reduction for the adaprobe model.
	"""
	npts = grid.shape[0]
	eed_grid = np.zeros(npts)
	for j, g in enumerate(grid):
		Hw = np.sum(np.log(beta))
		entrops = np.zeros(n_samples) 
		for i in range(n_samples):
			y = _get_sample(mu, beta, shape, rate, phi_map, phi_cov, omega, C, g, I)
			out = cavi_online(
				y, (g, I), mu, beta, shape, rate, phi_map, phi_cov, omega, C
			)
			next_beta = out[1]
			entrops[i] = np.sum(np.log(next_beta))

		eed_grid[j] = Hw - np.mean(entrops)
	return eed_grid

@njit
def _expected_entropy_decrease_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik, 
	n_samples):
	"""Calculate expected posterior entropy reduction for the spike-and-slab adaprobe model.
	"""
	Hw = np.sum(np.log(beta) - alpha * np.log(alpha + EPS) - (1 - alpha) * np.log(1 - alpha + EPS))
	entrops = np.zeros(n_samples)
	for i in range(n_samples):
		yk = _get_sample_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik)
		out = cavi_online_spike_and_slab(
			yk, (Lk, Ik), mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C
		)
		entrops[i] = np.sum(np.log(next_beta) - next_alpha * np.log(next_alpha + EPS) \
			- (1 - next_alpha) * np.log(1 - next_alpha + EPS))
	return Hw - np.mean(entrops)

@njit(parallel=True)
def _expected_entropy_decrease_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, grid, I, n_samples):
	"""Calculate expected posterior entropy reduction for the spike-and-slab adaprobe model in three dimensions.
	"""
	eed_grid = np.zeros(grid.shape[0])
	alpha = alpha.copy()
	alpha[alpha >= 1 - EPS] = 1 - EPS
	alpha[alpha <= EPS] = EPS
	for j in prange(len(grid)):
		g = grid[j]
		Hw = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) - alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha))
		entrops = np.zeros(n_samples)
		for i in prange(n_samples):
			y = _get_sample_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, g, I)
			out = cavi_online_spike_and_slab_3d_omega(
				y, (g, I), mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C
			)
			next_beta, next_alpha = out[1], out[2]
			next_alpha[next_alpha >= 1 - EPS] = 1 - EPS
			next_alpha[next_alpha <= EPS] = EPS
			entrops[i] = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * next_beta**2) - next_alpha * np.log(next_alpha) \
				- (1 - next_alpha) * np.log(1 - next_alpha))
		eed_grid[j] = Hw - np.mean(entrops)
	return eed_grid

@njit(parallel=True)
def _expected_entropy_decrease_spike_and_slab_on_grid(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, 
	grid, I, n_samples):
	"""Calculate expected posterior entropy reduction for the spike-and-slab adaprobe model.
	"""
	eed_grid = np.zeros(grid.shape[0])
	alpha = alpha.copy()
	alpha[alpha >= 1 - EPS] = 1 - EPS
	alpha[alpha <= EPS] = EPS
	for j in range(len(grid)):
		g = grid[j]
		Hw = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) - alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha))
		entrops = np.zeros(n_samples)
		for i in prange(n_samples):
			y = _get_sample_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, g, I)
			out = cavi_online_spike_and_slab(
				y, (g, I), mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C
			)
			next_beta, next_alpha = out[1], out[2]
			next_alpha[next_alpha >= 1 - EPS] = 1 - EPS
			next_alpha[next_alpha <= EPS] = EPS
			entrops[i] = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * next_beta**2) - next_alpha * np.log(next_alpha) \
				- (1 - next_alpha) * np.log(1 - next_alpha))
		eed_grid[j] = Hw - np.mean(entrops)
	return eed_grid

@njit(parallel=True)
def _expected_universal_entropy_decrease_on_grid(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, grid,
	I, n_samples):
	"""
	"""
	eed_grid = np.zeros(grid.shape[0])
	alpha = alpha.copy()
	alpha[alpha >= 1 - EPS] = 1 - EPS
	alpha[alpha <= EPS] = EPS
	N = len(alpha)

	# Calculate entropy of current posteriors
	Hw = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) \
		- alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha))\
		+ shape - np.log(rate) + scipy.special.loggamma(shape) + (1 - shape) * psi(shape)
	for n in range(N):
		Hw = Hw + 1/2 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * phi_cov[n]))
	
	# Simulate hypothetical responses
	for j in prange(len(grid)):
		g = grid[j]
		entrops = np.zeros(n_samples)
		
		for i in prange(n_samples):
			y = _get_sample_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, g, I)
			out = cavi_online_spike_and_slab(
				y, (g, I), mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C
			)
			_, next_beta, next_alpha, _, next_shape, next_rate, _, next_phi_cov = out
			next_alpha[next_alpha >= 1 - EPS] = 1 - EPS
			next_alpha[next_alpha <= EPS] = EPS

			entrops[i] = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * next_beta**2) \
				- next_alpha * np.log(next_alpha) - (1 - next_alpha) * np.log(1 - next_alpha)) \
				+ next_shape - np.log(next_rate) + scipy.special.loggamma(next_shape) + (1 - next_shape) * psi(next_shape)
			for n in range(N):
				entrops[i] = entrops[i] + 1/2 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * next_phi_cov[n]))
		eed_grid[j] = Hw - np.mean(entrops)
	return eed_grid

@njit(parallel=True)
def _expected_universal_entropy_decrease_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, grid, I, n_samples):
	"""Calculate expected posterior entropy reduction for the spike-and-slab adaprobe model in three dimensions.
	"""
	eed_grid = np.zeros(grid.shape[0])
	alpha = alpha.copy()
	alpha[alpha >= 1 - EPS] = 1 - EPS
	alpha[alpha <= EPS] = EPS
	N = len(mu)

	# Calculate entropy of current posteriors
	Hw = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * beta**2) \
		- alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha))\
		+ shape - np.log(rate) + scipy.special.loggamma(shape) + (1 - shape) * psi(shape)
	for n in range(N):
		Hw = Hw + 1/2 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * phi_cov[n]))

	# Simulate hypothetical responses
	for j in prange(len(grid)):
		g = grid[j]
		entrops = np.zeros(n_samples)
		for i in prange(n_samples):
			y = _get_sample_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, g, I)
			out = cavi_online_spike_and_slab_3d_omega(
				y, (g, I), mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C
			)
			# mu, beta, alpha, lamk, shape, rate, phi_map, phi_cov
			next_beta, next_alpha, next_shape, next_rate, next_phi_cov = out[1], out[2], out[4], out[5], out[7]
			next_alpha[next_alpha >= 1 - EPS] = 1 - EPS
			next_alpha[next_alpha <= EPS] = EPS

			entrops[i] = np.sum(1/2 * np.log(2 * np.pi * np.exp(1) * next_beta**2) \
				- next_alpha * np.log(next_alpha) - (1 - next_alpha) * np.log(1 - next_alpha)) \
				+ next_shape - np.log(next_rate) + scipy.special.loggamma(next_shape) + (1 - next_shape) * psi(next_shape)
			for n in range(N):
				entrops[i] = entrops[i] + 1/2 * np.log(np.linalg.det(2 * np.pi * np.exp(1) * next_phi_cov[n]))

		eed_grid[j] = Hw - np.mean(entrops)
	return eed_grid

@njit
def _get_sample(mu, beta, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik):
	N = mu.shape[0]
	phi = np.zeros((N, 2))
	for n in range(N):
		phi[n] = _sample_phi(phi_map[n], phi_cov[n])
	fk = sigmoid(phi[:, 0] * Ik * np.exp(-omega * np.sum(np.square(Lk - C), 1)) - phi[:, 1])
	sk = (np.random.rand(N) <= fk).astype(float64)
	u = beta * np.random.standard_normal(N) + mu
	sig = np.sqrt(1/(np.random.gamma(shape, 1/rate)))
	y = np.sum(u * sk) + np.sqrt(sig) * np.random.standard_normal()
	return y

@njit
def _get_sample_spike_and_slab(mu, beta, alpha, shape, rate, phi_map, phi_cov, omega, C, Lk, Ik):
	N = mu.shape[0]
	phi = np.zeros((N, 2))
	for n in range(N):
		phi[n] = _sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)
	fk = sigmoid(phi[:, 0] * Ik * np.exp(-omega * np.sum(np.square(Lk - C), 1)) - phi[:, 1])
	sk = (np.random.rand(N) <= fk).astype(float64)
	gam = (np.random.rand(N) <= alpha).astype(float64)
	u = beta * np.random.standard_normal(N) + mu
	sig = np.sqrt(1/(np.random.gamma(shape, 1/rate)))
	y = np.sum(gam * u * sk) + sig * np.random.standard_normal()
	return y

@njit
def _get_sample_spike_and_slab_3d(mu, beta, alpha, shape, rate, phi_map, phi_cov, Omega, C, Lk, Ik):
	N = mu.shape[0]
	phi = np.zeros((N, 2))
	fk = np.zeros((N))
	for n in range(N):
		phi[n] = _sample_phi_independent_truncated_normals(phi_map[n], phi_cov[n], num_mc_samples=1)
		fk[n] = sigmoid(phi[n, 0] * Ik * np.exp(-(Lk - C[n]) @ Omega[n] @ (Lk - C[n])) - phi[n, 1])
	sk = (np.random.rand(N) <= fk).astype(float64)
	gam = (np.random.rand(N) <= alpha).astype(float64)
	u = beta * np.random.standard_normal(N) + mu
	sig = np.sqrt(1/(np.random.gamma(shape, 1/rate)))
	y = np.sum(gam * u * sk) + sig * np.random.standard_normal()
	return y
