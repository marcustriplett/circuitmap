import numpy as np
import time

def cosamp(A, y, k, tol=1e-8, maxiter=500, x=None):
	'''Compressive sampling matching pursuit (CoSaMP) algorithm.

	**** This module is duplicated from the mr_utils package by N. McKibben (2019).

	Please see `https://github.com/mckib2/mr_utils/blob/master/mr_utils/cs/greedy/cosamp.py` 
	for the original implementation. ****


	Parameters
	==========
	A : array_like
		Measurement matrix.
	y : array_like
		Measurements (i.e., y = Ax).
	k : int
		Number of expected nonzero coefficients.
	lstsq : {'exact', 'lm', 'gd'}, optional
		How to solve intermediate least squares problem.
	tol : float, optional
		Stopping criteria.
	maxiter : int, optional
		Maximum number of iterations.
	x : array_like, optional
		True signal we are trying to estimate.
	disp : bool, optional
		Whether or not to display iterations.

	Returns
	=======
	x_hat : array_like
		Estimate of x.

	Notes
	=====
	lstsq function
	- 'exact' solves it using numpy's linalg.lstsq method.
	- 'lm' uses solves with the Levenberg-Marquardt algorithm.
	- 'gd' uses 3 iterations of a gradient descent solver.

	Implements Algorithm 8.7 from [1]_.

	References
	==========
	.. [1] Eldar, Yonina C., and Gitta Kutyniok, eds. Compressed sensing:
		   theory and applications. Cambridge University Press, 2012.
	
	'''
	t_start = time.time()

	# length of measurement vector and original signal
	_n, N = A.shape[:]

	# Initializations
	x_hat = np.zeros(N, dtype=y.dtype)
	r = y.copy()
	ynorm = np.linalg.norm(y)

	if x is None:
		x = np.zeros(x_hat.shape, dtype=y.dtype)
	elif x.size < x_hat.size:
		x = np.hstack(([0], x))

	lstsq_fun = lambda A0, y: np.linalg.lstsq(A0, y, rcond=None)[0]

	for ii in range(maxiter):

		# Get step direction
		g = np.dot(A.conj().T, r)

		# Add 2*k largest elements of g to support set
		Tn = np.union1d(x_hat.nonzero()[0], np.argsort(np.abs(g))[-(2*k):])

		# Solve the least squares problem
		xn = np.zeros(N, dtype=y.dtype)
		xn[Tn] = lstsq_fun(A[:, Tn], y)

		xn[np.argsort(np.abs(xn))[:-k]] = 0
		x_hat = xn.copy()

		# Compute new residual
		r = y - np.dot(A, x_hat)

		# Compute stopping criteria
		stop_criteria = np.linalg.norm(r)/ynorm

		# Check stopping criteria
		if stop_criteria < tol:
			break

	t_end = time.time()

	return x_hat, t_end - t_start