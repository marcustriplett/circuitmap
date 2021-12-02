import numpy as np
from scipy.special import ndtri, ndtr
from prettytable import PrettyTable
import _pickle as cpickle # pickle compression
import pandas as pd
import bz2
import os

def sample_truncnorm(mean, sdev, size=1):
	u = np.random.uniform(0, 1, size)
	return ndtri(ndtr(-mean/sdev) + u * (1 - ndtr(-mean/sdev))) * sdev + mean

def sigmoid(x):
	return 1/(1 + np.exp(-x))

class CrossValidation:
	'''Basic class for storing, loading, updating, and displaying cross-validation statistics.
	'''
	def __init__(self, nfolds, params, vals):
		'''Initialise CrossValidation object
		'''
		self.nfolds = nfolds
		self.folds = [None for _ in range(nfolds)]
		self.params = params
		self.vals = vals
		self.stats = {'mean': None, 'std': None}

	def __str__(self):
		summary = PrettyTable()
		summary.title = f'Cross-validation record for parameter(s) {self.params} with value(s) {self.vals} (log pointwise predictive density).'
		summary.field_names = ['Num folds'] + ['Fold %i'%(fold+1) for fold in range(self.nfolds)] \
			+ ['Mean', 'Std']
		entry = [self.nfolds] + ['%.2f'%fold['log_pointwise_predictive_density'] for fold in self.folds] \
			+ ['%.2f'%self.stats['mean'], '%.2f'%self.stats['std']]
		summary.add_row(entry)

		return summary.get_string()

	def update(self, fold=None, test_obs=None, test_stim=None, predictive_distribution=None, lppd=None):
		'''Update cross-validation record.
		'''
		self.folds[fold] = {
			'test_observation': test_obs,
			'test_stimulus': test_stim,
			'predictive_distribution': predictive_distribution,
			'log_pointwise_predictive_density': lppd
		}
		
		lppds = [f['log_pointwise_predictive_density'] for f in self.folds if f is not None]

		self.stats['mean'], self.stats['std'] = np.mean(lppds), np.std(lppds)

	def save(self, path):
		# if path[-1] != '/': path += '/'
		with bz2.BZ2File(path + f'CV_param_{self.params}_val_{self.vals}_nfolds_{self.nfolds}.pkl', 'wb') as savefile:
			cpickle.dump(self, savefile)

def load_CV(path):
	with bz2.BZ2File(path, 'rb') as pkl_file:
		cv = cpickle.load(pkl_file)
	return cv

def load_CV_dir(fdir):
	if fdir[-1] != '/': fdir += '/'
	files = os.listdir(fdir)
	num_files = len(files)
	cv = load_CV(fdir + files[0])
	df = pd.DataFrame(columns=[p for p in cv.params] + ['mean_lppd'] + ['fold %i'%i for i in range(cv.nfolds)])
	for f in files:
		cv = load_CV(fdir + f)
		folds = [fold['log_pointwise_predictive_density'] for fold in cv.folds]
		row = {p: val for p, val in zip(cv.params, cv.vals)}
		row.update({'mean_lppd': cv.stats['mean']})
		row.update({('fold %i'%i): fold for i, fold in enumerate(folds)})
		df = df.append(row, ignore_index=True)

	# Return optimal sigma if performing model selection else complete dataframe
	return df

