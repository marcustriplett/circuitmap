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
	def __init__(self, nfolds, param, val):
		'''Initialise CrossValidation object
		'''
		self.nfolds = nfolds
		self.folds = [None for _ in range(nfolds)]
		self.param = param
		self.val = val
		self.stats = {'mean': None, 'std': None}

	def __str__(self):
		summary = PrettyTable()
		summary.title = f'Cross-validation record for parameter "{self.param}" with value "{self.val:.2f}".\nStatistic: log pointwise predictive density.'
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
		with bz2.BZ2File(path + 'CV_param_%s_val_%.3f_nfolds_%i.pkl'%(self.param, self.val, self.nfolds), 'wb') as savefile:
			cpickle.dump(self, savefile)

def load_CV(path):
	with bz2.BZ2File(path, 'rb') as pkl_file:
		cv = cpickle.load(pkl_file)
	return cv

def load_CV_dir(fdir, select=False):
	if fdir[-1] != '/': fdir += '/'
	files = os.listdir(fdir)
	num_files = len(files)
	df = pd.DataFrame(columns=['sigma', 'lppd'])
	for f in files:
		cv = load_CV(fdir + f)
		df = df.append({'sigma': cv.val, 'lppd': cv.stats['mean']}, ignore_index=True)

	# Return optimal sigma if performing model selection else complete dataframe
	if select:
		return df.groupby('sigma').mean().idxmax()[0]
	else:
		return df