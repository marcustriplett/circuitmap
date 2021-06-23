import numpy as np
from numba import njit, bool_

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

def get_psf_func(grid, z0=63, omega0=13, I_th=18):
	"""Generate PSF.
	"""
	def S_0(r, z, z0, omega0):
		return jnp.exp(-2*r**2/omega0**2) * jnp.exp(-2*z**2/z0**2)

	def S_S(r, z, I0, I_th, z0, omega0):
		alpha = jax.nn.relu(I0**2 - I_th**2)/I0**2
		zS = jnp.sqrt(alpha * z0**2 + 1e-8)
		omegaS = jnp.sqrt(alpha * omega0**2 + 1e-8)
		return jnp.exp(-2*r**2/omegaS**2) * jnp.exp(-2*z**2 / zS**2)

	def Se(r, z, I, I_th=I_th, z0=z0, omega0=omega0):
		Is = jnp.sqrt(jax.nn.relu(I**2 - I_th**2))
		return (I**2 * S_0(r, z, z0, omega0) - (I > I_th) * Is**2 * S_S(r, z, I, I_th, z0, omega0))/I_th**2

	def dist_euclid(vec1, vec2):
		return jnp.sqrt(jnp.sum(jnp.square(vec1 - vec2), 1))

	def dist_euclid_1d(vec1, vec2):
		return jnp.abs(vec1 - vec2)

	def _psf(g, I, center, I_th=I_th, z0=z0, omega0=omega0):
		return Se(dist_euclid(g[:, :2], center[:2]), dist_euclid_1d(g[:, -1], center[-1]), I, z0=z0, omega0=omega0, I_th=I_th)
	
	return jit(vmap(partial(_psf, grid), in_axes=(0, 0)))

def get_filt_grid_around_loc_multi(cell_locs, dim=20, num_points=4):
	return np.array([get_filt_grid_around_loc(loc, dim=dim, num_points=num_points) for loc in cell_locs])

def get_filt_grid_around_loc(loc, dim=20, num_points=4):
    xr, yr, zr = [np.linspace(loc[i] - dim/2, loc[i] + dim/2, num_points) for i in range(3)]
    xgrid, ygrid, zgrid = np.meshgrid(xr, yr, zr)
    return np.c_[xgrid.flatten(), ygrid.flatten(), zgrid.flatten()]

def get_gauss_func(grid):
    def _gauss(g, mu, Sigma):
        return jnp.exp(-(g - mu) @ Sigma @ (g - mu))
    return jit(partial(vmap(_gauss, in_axes=(0, None, None)), grid))

def nrelu(x):
	return x * (x > 0)

@njit
def sigmoid(x):
	return 1./(1. + np.exp(-x))

@njit
def get_mask(N):
	arr = np.ones((N, N))
	np.fill_diagonal(arr, 0)
	return arr.astype(bool_)

@njit
def soften(x):
	return (1 - 1e-8) * x + 1e-10