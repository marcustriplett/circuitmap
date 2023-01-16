import numpy as np
import circuitmap as cm
from circuitmap import NeuralDemixer
from sklearn.metrics import r2_score
import argparse
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import h5py

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def _get_pairwise_dist_xy(tars):
	return np.array([[np.sqrt(np.sum(np.square(tar1 - tar2))) for tar1 in tars] for tar2 in tars])

def _get_pairwise_adjacency_z(tars, planes):
	plane_representation = np.array([np.where(tar == planes)[0][0] for tar in tars])
	adj = np.array([[
		np.abs(tar1 - tar2) <= 1 for tar1 in plane_representation] 
		for tar2 in plane_representation
	]).astype(int)
	return adj

def _get_cluster_reps_pixelbased(clusters, targets, img):
	planes = np.unique(targets[:, -1])
	n_clusters = len(clusters)
	cluster_reps = [None for _ in range(n_clusters)]
	for i in range(n_clusters):
		pixel_brightness = []
		for cl in clusters[i]:
			tar = targets[cl].astype(int)
			depth_indx = np.where(tar[-1] == planes)[0][0]
			pixel_brightness += [img[0][depth_indx][tar[0], tar[1]]]
		cluster_reps[i] = clusters[i][np.argmax(pixel_brightness)]
	return cluster_reps

def compute_ridge_waveforms(psc, model, stim_matrix):
	cnx = np.where(model.state['mu'])[0]
	locs = np.unique(np.concatenate(np.array([np.where(stim_matrix[n])[0] for n in cnx])))
	lr = Ridge(fit_intercept=False, alpha=1e-3, positive=True)
	lr.fit(model.state['lam'][cnx][:, locs].T, psc[locs])
	return lr.coef_.T

def merge_duplicates(psc, stim_matrix, model, targets, img, mse_threshold=0.1, dist_threshold=15):
	planes = np.unique(targets[:, -1])
	weights = model.state['mu']
	found_cnx = np.where(weights)[0]
	n_cnx = len(found_cnx)
	waveforms = compute_ridge_waveforms(psc, model, stim_matrix)
	pairwise_errs = np.array([
		[
			np.sum(np.square(waveforms[cnx1] - waveforms[cnx2])) for cnx1 in range(n_cnx)
		] for cnx2 in range(n_cnx)
	])
	pairwise_adj = _get_pairwise_adjacency_z(targets[found_cnx][:, -1], planes)
	pairwise_close = (_get_pairwise_dist_xy(targets[found_cnx][:, :2]) < dist_threshold) * pairwise_adj # close in xy and lie on adjacent planes
	pairwise_duplicates = (pairwise_errs < mse_threshold) * pairwise_close
	clusters = [list(x) for x in set([tuple(found_cnx[np.where(row)[0]].tolist()) for row in pairwise_duplicates])] # extract duplicate clusters
	cluster_reps = _get_cluster_reps_pixelbased(clusters, targets, img) # select cluster representatives
	return cluster_reps

def lookup(coords, arr):
	return np.intersect1d(*[np.where(arr[:, i] == coords[i])[0] for i in range(2)])[0]

def plot_spike_inference_comparison(den_pscs, stim_matrices, models, spks=None, titles=None, save=None, 
	ymax=1.1, n_plots=15, max_trials_to_show=30, col_widths=None, row_height=0.6, order=None, trial_len=900):
	if col_widths is None:
		col_widths = 7.5 * np.ones(len(models))
		
	N = stim_matrices[0].shape[0]
	Is = [np.array([np.unique(stim[:, k])[1] for k in range(stim.shape[1])])
		 for stim in stim_matrices]
	ncols = len(models)
	
	fig = plt.figure(figsize=(np.sum(col_widths), row_height * n_plots * 1.5))
	gs = fig.add_gridspec(ncols=ncols, nrows=n_plots, hspace=0.5, wspace=0.05, width_ratios=col_widths/col_widths[0])
	
	normalisation_factor = np.max(np.abs(np.vstack(den_pscs)))
	mu_norm = np.max(np.abs([model.state['mu'] for model in models]))
	ymin = -0.05 * ymax
	
	trace_linewidth = 0.65
	
	if order is None:
		order = np.argsort(models[0].state['mu'])[::-1]
	
	for col in range(ncols):
		for m in range(n_plots):
			n = order[m]

			# spike predictions
			ax = fig.add_subplot(gs[m, col])
			if m == 0 and titles is not None:
				plt.title(titles[col], fontsize=fontsize, y=1.5)

			powers = np.unique(Is[col])
			trials_per_power = max_trials_to_show // len(powers)
			stim_locs = np.array([])
			for pwr in powers:
				stim_locs = np.concatenate([stim_locs, np.where(stim_matrices[col][n] == pwr)[0][:trials_per_power]])

			stim_locs = stim_locs.astype(int)
			this_y_psc = den_pscs[col][stim_locs].flatten()/normalisation_factor
			n_repeats = np.min([len(stim_locs), max_trials_to_show])
			trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)

			plt.xlim([0, trial_len*n_repeats])
			facecol = 'firebrick'
			model = models[col]
			lam = model.state['lam']
			K = lam.shape[1]
			mu = model.state['mu'].copy()
			trace_col = 'k' if mu[n] != 0 else 'gray'
			
			if 'z' in model.state.keys():
				z = model.state['z']
			else:
				z = np.zeros(K)
				
			for tb in range(len(trial_breaks) - 1):
				if tb > 0:
					plt.plot([trial_breaks[tb], trial_breaks[tb]], [ymin, ymax], '--', color=trace_col)
					
				ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=facecol, 
								 edgecolor='None', alpha=lam[n, stim_locs][tb] * 0.5, zorder=-5)

				# Plot power changes
				if (m == 0) and (Is[col][stim_locs][tb] != Is[col][stim_locs][tb-1]):
					plt.text(trial_breaks[tb], 1.1 * ymax, '%i mW'%Is[col][stim_locs][tb], fontsize=fontsize-2)
				   
				if z[stim_locs][tb] != 0:
					plt.plot(trial_len * (tb + 0.5), 0.7 * ymax, marker='*', markerfacecolor='b', markeredgecolor='None', markersize=6)

			plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)

			for loc in ['top', 'right', 'left', 'bottom']:
				plt.gca().spines[loc].set_visible(False)
			plt.xticks([])
			plt.yticks([])
			plt.ylim([ymin, ymax])
			
			if col == 0:
				label_col = 'k'
				plt.ylabel(m+1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color=label_col)

				fig.supylabel('Neuron', fontsize=fontsize, x=0.09) #x=0.0825)
			ax.set_rasterization_zorder(-2)
	
	if save is not None:
		plt.savefig(save, format='png', bbox_inches='tight', dpi=400, facecolor='white')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--demixer')
	parser.add_argument('--msrmp')
	parser.add_argument('--out')
	parser.add_argument('--fmt')
	parser.add_argument('--reader')
	parser.add_argument('--sign')
	args = parser.parse_args()

	print('Loading file at ', args.data)
	if args.reader == 'h5py':
		data = h5py.File(args.data)
		stim_matrix = np.array(data['stimulus_matrix']).T
		psc = np.array(data['pscs']).T
		targets = np.array(data['targets']).T
		planes = np.unique(targets[:, -1])
		img = np.array([np.array(data[data['img'][i][0]]) for i in range(len(planes))])
	elif args.reader == 'scipy':
		data = loadmat(args.data)
		stim_matrix = data['stimulus_matrix']
		psc = data['pscs']
		targets = data['targets']
		planes = np.unique(targets[:, -1])
		img = data['img']
	else:
		raise Exception

	fmt = 'mat' if args.fmt is None else args.fmt
	print('Data attributes:')
	print('\tPSC array dimension ', psc.shape)
	print('\tStimulus matrix dimension ', stim_matrix.shape)
	print('\tImage dimension ', img.shape)
	print('\tData modality ', args.sign)
	print('Save format ', fmt)

	demix = NeuralDemixer(path=args.demixer, device='cpu')

	N, K = stim_matrix.shape
	msrmp = float(args.msrmp)

	model_single, model_multi = cm.Model(N), cm.Model(N)

	single_tar_locs = np.where(np.sum(stim_matrix > 0, 0) == 1)[0]
	multi_tar_locs = np.where(np.sum(stim_matrix > 0, 0) > 1)[0]

	print('Number (single target, ensemble) trials (%i, %i)'%(len(single_tar_locs), len(multi_tar_locs)))

	if (len(single_tar_locs) == 0) or (len(multi_tar_locs) == 0):
		print('At least one dataset empty, cancelling job.')
		exit()

	stim_single = stim_matrix[:, single_tar_locs] * 1.0
	stim_multi = stim_matrix[:, multi_tar_locs] * 1.0

	if args.sign == 'negative':
		psc = -psc

	psc_dem = demix(psc)
	psc_dem_single, psc_dem_multi = psc_dem[single_tar_locs], psc_dem[multi_tar_locs]

	print('Fitting models...')
	model_single.fit(psc_dem_single, stim_single, method='caviar', fit_options={'msrmp': msrmp, 'save_histories': False})
	model_multi.fit(psc_dem_multi, stim_multi, method='caviar', fit_options={'msrmp': msrmp, 'save_histories': False})
	print('Complete.')

	# cnx_single = merge_duplicates(psc_dem_single, stim_single, model_single, targets, img)
	# cnx_multi = merge_duplicates(psc_dem_multi, stim_multi, model_multi, targets, img)
	cnx_single = np.where(model_single.state['mu'])
	cnx_multi = np.where(model_multi.state['mu'])

	#% PLOTTING
	ms = 20
	spike_threshold = 0.1
	spk_start = 100
	spk_end = 240
	stim_upper = 0.6
	stim_lower = -0.6
	fontsize = 14
	xrange = np.arange(-100, 800)

	minv, maxv = [func(np.concatenate([model_single.state['mu'], model_multi.state['mu']])) for func in [np.min, np.max]]
	powers = np.unique(stim_matrix)[1:].astype(float)
	npowers = len(powers)

	figure_mosaic = """
		12345
		ABCDE
		6666F
	"""

	fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(15, 12))
	# for i in range(5):
	# 	ax = axes[str(i + 1)]
	# 	ax.imshow(data['img'][0, i])
	# 	ax.set_title(str(int(planes[i])) + ' $\mu$m', fontsize=fontsize)
	# 	ax.set_xticks([])
	# 	ax.set_yticks([])
	# 	if i == 0:
	# 		ax.set_ylabel('Single-target\nconnections', fontsize=fontsize)
	# 	targets_cnx = targets[cnx_single]
	# 	tars = targets_cnx[targets_cnx[:, -1] == planes[i]][:, :2]
	# 	ax.scatter(tars[:, 1], tars[:, 0], edgecolor='white', facecolor='None', marker='o', s=13, linewidth=0.5)
	# 	for j, tar in enumerate(tars):
	# 		ax.text(tars[j, 1], tars[j, 0], lookup(tars[j], targets), color='white', fontsize=8)
			
	# for i, st in enumerate(['A', 'B', 'C', 'D', 'E']):
	# 	ax = axes[st]
	# 	ax.imshow(data['img'][0, i])
	# 	ax.set_xticks([])
	# 	ax.set_yticks([])
	# 	if i == 0:
	# 		ax.set_ylabel('Ten-target\nconnections', fontsize=fontsize)

	# 	targets_cnx = targets[cnx_multi]
	# 	tars = targets_cnx[targets_cnx[:, -1] == planes[i]]
	# 	ax.scatter(tars[:, 1], tars[:, 0], edgecolor='white', facecolor='None', marker='o', s=13, linewidth=0.5)
	# 	for j, tar in enumerate(tars):
	# 		ax.text(tars[j, 1], tars[j, 0], lookup(tars[j], targets), color='white', fontsize=8)

	ax = axes['6']
	for n in range(N):
		if n == 0:
			label = 'Single target'
		else:
			label = None
		ax.plot([n, n], [0, model_single.state['mu'][n]], color='k', label=label, zorder=100)
	ax.errorbar(np.arange(N), model_multi.state['mu'], yerr=model_multi.state['beta']*(model_multi.state['mu'] != 0), color='firebrick', 
				 zorder=-1, label='Ten target', fmt='.', mfc='white', markersize=12, capsize=2, mew=2)
	ax.legend(fontsize=fontsize)
	ax.set_xlabel('Neurons', fontsize=fontsize)
	ax.set_ylabel('Synaptic weight (nC)', fontsize=fontsize)
	ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=fontsize)
	ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=fontsize)
	ax.set_xlim([-1, N+1])

	ax = axes['F']
	ax.scatter(model_single.state['mu'], model_multi.state['mu'], edgecolor='k', facecolor='w', zorder=100)
	minv, maxv = [func(np.concatenate([model_single.state['mu'], model_multi.state['mu']])) for func in [np.min, np.max]]
	ax.plot([minv, maxv], [minv, maxv], '--', color='gray', zorder=-1)
	ax.set_xlabel('Single target', fontsize=fontsize)
	ax.set_ylabel('Ten target', fontsize=fontsize)
	ax.set_xticks(np.arange(0, maxv + 1, 20))
	ax.set_yticks(np.arange(0, maxv + 1, 20))
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.set_ylim([minv - 2, maxv + 2])
	ax.set_xlim([minv - 2, maxv + 2])

	out = args.out
	if out[-1] != '/': out += '/'
	fn = args.data.split('/')[-1][:-4] # extract filename and strip ext
	# fig.savefig('%s_%s_msrmp%s_summary.pdf'%(args.out, fn, args.msrmp), format='pdf', bbox_inches='tight', facecolor='white', dpi=400)
	fig.savefig('%s%s_msrmp%s_summary.png'%(out, fn, args.msrmp), format='png', bbox_inches='tight', facecolor='white', dpi=400)

	psc_dems = [psc_dem_single, psc_dem_multi]
	stim_matrices = [stim_single, stim_multi]
	models = [model_single, model_multi]
	titles = ['Single target', 'Ten target']

	plot_spike_inference_comparison(psc_dems, stim_matrices, models, titles=titles, ymax=1.1, n_plots=30, max_trials_to_show=60,
								col_widths=np.array([7, 14]), row_height=0.6, order=None, trial_len=900,
								save='%s%s_msrmp%s_checkerboard.png'%(out, fn, args.msrmp))

	savename = '%s%s_msrmp%s_models'%(out, fn, args.msrmp)
	print('Saving file ', savename + '.' + fmt)

	if fmt == 'npz':
		np.savez(savename + '.' + fmt, weights_single=model_single.state['mu'], weight_uncertainty_single=model_single.state['beta'],
			weights_ensemble=model_multi.state['mu'], weight_uncertainty_ensemble=model_multi.state['beta'],
			spikes_single=model_single.state['lam'], spikes_ensemble=model_multi.state['lam'])
	elif fmt == 'mat':
		savemat(savename + '.' + fmt, {'weights_single': model_single.state['mu'], 'weight_uncertainty_single': model_single.state['beta'],
			'weights_ensemble': model_multi.state['mu'], 'weight_uncertainty_ensemble': model_multi.state['beta'],
			'spikes_single': model_single.state['lam'], 'spikes_ensemble': model_multi.state['lam']})
