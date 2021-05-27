import numpy as np
from numba import njit, bool_

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