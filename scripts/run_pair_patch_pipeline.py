import numpy as np
import circuitmap as cm
from circuitmap import NeuralDemixer
from sklearn.metrics import r2_score
import argparse
from scipy.io import loadmat

def lookup(coords, arr):
	return np.intersect1d(*[np.where(arr[:, i] == coords[i])[0] for i in range(2)])[0]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data')
	parser.add_argument('--demixer')
	parser.add_argument('--msrmp')
	parser.add_argument('--out')
	args = parser.parse_args()

	print('Loading file at ', args.data)
	data = loadmat(args.data)
	stim_matrix = data['stimulus_matrix']
	psc = data['pscs']
	psp = data['psps']
	targets = data['targets']

	demix = NeuralDemixer(path=args.demixer, device='cpu')

	N = targets.shape[0]
	msrmp = float(args.msrmp)

	model_single, model_multi = cm.Model(N), cm.Model(N)

	single_tar_locs = np.where(np.sum(stim_matrix > 0, 0) == 1)[0]
	multi_tar_locs = np.where(np.sum(stim_matrix > 0, 0) > 1)[0]

	stim_single = stim_matrix[:, single_tar_locs]
	stim_multi = stim_matrix[:, multi_tar_locs]

	psc_single, psc_multi = [demix(arr) for arr in [psc[single_tar_locs], psc[multi_tar_locs]]]

	print('Fitting models...')
	for model, psc, stim in zip([model_single, model_multi], [psc_single, psc_multi], [stim_single, stim_multi]):
		model.fit(psc, stim, method='caviar', fit_options={'minimax_spk_prob': msrmp})
	print('Complete.')

	#% PLOTTING

	minv, maxv = [func(np.concatenate([model_single.state['mu'], model_multi.state['mu']])) for func in [np.min, np.max]]
	powers = np.unique(stim_matrix)[1:].astype(float)
	npowers = len(powers)
	spike_threshold = 0.1
	spk_start = 100
	spk_end = 240
	presyn_spikes = np.zeros(N)
	for n in range(N):
		locs = np.where(stim_matrix[n])[0]
		presyn_spikes[n] = len(np.where(np.max(np.abs(psp[locs, spk_start:spk_end]), axis=1) > spike_threshold)[0])
	lp_cell = np.argmax(presyn_spikes)

	# Extract LP spikes and power curves
	spks = np.zeros(K)
	locs = np.where(stim_matrix[lp_cell])[0]
	spks[locs] = np.max(np.abs(psp[locs, spk_start:spk_end]), axis=1) > spike_threshold

	frates_single_tar, frates_std_single_tar, frates_multi_tar, frates_std_multi_tar = [
		np.zeros(npowers) for _ in range(4)
	]

	lam_single_tar, lam_std_single_tar, lam_multi_tar, lam_std_multi_tar = [
		np.zeros(npowers) for _ in range(4)
	]

	for i, p in enumerate(powers):
		stim_locs = np.intersect1d(np.where(stim_matrix[lp_cell] == p)[0], single_tar_locs)
		frates_single_tar[i] = np.mean(spks[stim_locs])
		frates_std_single_tar[i] = np.std(spks[stim_locs])
		lam_single_tar[i] = np.mean(model_single.state['lam'][lp_cell, np.where(stim_single[lp_cell] == p)[0]])
		lam_std_single_tar[i] = np.std(model_single.state['lam'][lp_cell, np.where(stim_single[lp_cell] == p)[0]])

	for i, p in enumerate(powers):
		stim_locs = np.intersect1d(np.where(stim_matrix[lp_cell] == p)[0], multi_tar_locs)
		frates_multi_tar[i] = np.mean(spks[stim_locs])
		frates_std_multi_tar[i] = np.std(spks[stim_locs])
		lam_multi_tar[i] = np.mean(model_multi.state['lam'][lp_cell, np.where(stim_multi[lp_cell] == p)[0]])
		lam_std_multi_tar[i] = np.std(model_multi.state['lam'][lp_cell, np.where(stim_multi[lp_cell] == p)[0]])

	ms = 20
	stim_upper = 0.6
	stim_lower = -0.6
	figure_mosaic = """
		12345
		ABCDE
		6666F
		77889
	"""

	fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(15, 12))
	for i in range(5):
		ax = axes[str(i + 1)]
		ax.imshow(data['img'][0, i])
		ax.set_title(str(int(planes[i])) + ' $\mu$m', fontsize=fontsize)
		ax.set_xticks([])
		ax.set_yticks([])
		if i == 0:
			ax.set_ylabel('Single-target\nconnections', fontsize=fontsize)
		found_cnx = np.where(model_single.state['mu'])[0]
		targets_cnx = targets[found_cnx]
		tars = targets_cnx[targets_cnx[:, -1] == planes[i]][:, :2]
		ax.scatter(tars[:, 1], tars[:, 0], edgecolor='white', facecolor='None', marker='o', s=13, linewidth=0.5)
		for j, tar in enumerate(tars):
			ax.text(tars[j, 1], tars[j, 0], lookup(tars[j], targets), color='white', fontsize=8)
			
	for i, st in enumerate(['A', 'B', 'C', 'D', 'E']):
		ax = axes[st]
		ax.imshow(data['img'][0, i])
		ax.set_xticks([])
		ax.set_yticks([])
		if i == 0:
			ax.set_ylabel('Ten-target\nconnections', fontsize=fontsize)

		found_cnx = np.where(model_multi.state['mu'])[0]
		targets_cnx = targets[found_cnx]
		tars = targets_cnx[targets_cnx[:, -1] == planes[i]]
		ax.scatter(tars[:, 1], tars[:, 0], edgecolor='white', facecolor='None', marker='o', s=13, linewidth=0.5)
		for j, tar in enumerate(tars):
			ax.text(tars[j, 1], tars[j, 0], lookup(tars[j], targets), color='white', fontsize=8)

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

	ax = axes['7']
	locs = np.where(stim_single[lp_cell])[0]
	ax.set_title('Single target', fontsize=fontsize)
	ax.plot(xrange, psp[locs].T, color='k', zorder=50, linewidth=0.5)
	ax.plot(xrange, psc[locs].T, color='gray', zorder=0, linewidth=0.5)
	ax.plot(xrange, psc_dem[locs].T, zorder=100)
	ax.plot([-100, 800], [-spike_threshold, -spike_threshold], '--', color='gray')
	ax.fill_between([0, 60], [stim_lower, stim_lower], [stim_upper, stim_upper], facecolor='firebrick', alpha=0.25, edgecolor='None')
	ax.set_xticks(np.arange(0, 41 * ms, 10 * ms))
	ax.set_xticklabels(np.arange(0, 41, 10))
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.set_xlabel('Time (ms)', fontsize=fontsize)

	ax = axes['8']
	locs = np.intersect1d(np.where(stim_matrix[lp_cell])[0], multi_tar_locs)
	ax.set_title('Multi target', fontsize=fontsize)
	ax.plot(xrange, psp[locs].T, color='k', zorder=50, linewidth=0.5)
	ax.plot(xrange, psc[locs].T, color='gray', zorder=0, linewidth=0.5)
	ax.plot(xrange, psc_dem[locs].T, zorder=100)
	ax.plot([-100, 800], [-spike_threshold, -spike_threshold], '--', color='gray')
	ax.fill_between([0, 60], [stim_lower, stim_lower], [stim_upper, stim_upper], facecolor='firebrick', alpha=0.25, edgecolor='None')
	ax.set_xticks(np.arange(0, 41 * ms, 10 * ms))
	ax.set_xticklabels(np.arange(0, 41, 10))
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.set_xlabel('Time (ms)', fontsize=fontsize)

	ax = axes['9']
	ax.errorbar(powers, frates_single_tar, frates_std_single_tar, marker='o', color='k', label='LP single')
	ax.errorbar(powers, lam_single_tar, lam_std_single_tar, marker='o', linestyle='--', color='gray', label='Model single')
	ax.errorbar(powers, frates_multi_tar, frates_std_multi_tar, marker='o', color='navy', label='LP Multi')
	ax.errorbar(powers, lam_multi_tar, lam_std_multi_tar, marker='o', linestyle='--', color='C0', label='Model multi')
	ax.set_xlabel('Power', fontsize=fontsize)
	ax.set_ylabel('Spike prob.', fontsize=fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.legend(ncol=1)
	fig.tight_layout()

	out = args.out
	if out[-1] != '/': out += '/'
	fn = args.data.split('/')[-1][:-4] # extract filename and strip ext
	fig.savefig('%s_%s_msrmp%s.pdf'%(args.out, fn, args.msrmp), format='pdf', bbox_inches='tight', facecolor='white', dpi=400)
	fig.savefig('%s_%s_msrmp%s.png'%(args.out, fn, args.msrmp), format='png', bbox_inches='tight', facecolor='white', dpi=400)

