import numpy as np
from scipy.sparse import csc_matrix
from sklearn.linear_model import Lasso as skl_lasso

def lasso(targets, stimuli, y, ntrials):
	"""For fixed plane, power.
	"""
	nstim = len(targets)
	row = np.array(range(nstim * ntrials))
	col = np.array(list(range(nstim)) * ntrials)
	sp_data = np.ones(len(col))

	A = csc_matrix((sp_data, (row, col)))
	alpha = 1e-4
	sparse_lasso = skl_lasso(alpha=alpha, fit_intercept=False, max_iter=100)
	sparse_lasso.fit(A, y)
	
	b = sparse_lasso.coef_
	return b