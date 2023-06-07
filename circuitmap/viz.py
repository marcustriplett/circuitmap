import matplotlib.pyplot as plt
import numpy as np

def plot_checkerboard(_psc, _stim_matrix, model, true_spikes=None, true_weights=None, spike_thresh=0.01, save=None, ymax=None, n_plots=15, max_trials_to_show=30, 
	col_width=7.5, row_height=0.6, order=None, sdevs=None, fig_width=None, overlay_spikes=False, annotate_spikes=False, wspace=0.05, labels=None,
	hspace=0.5, ylabelpad=0.05, facecol=None, edgecol=None, trial_len=900, save_fmt='png', fontsize=14, append_last_row=False, backend='pgf', plot_sponts=True,
	spont_alpha=0.75, spont_col='C0', trials=None):
	''' plot_checkerboard
	'''

	# Parse trial subset
	if trials is None:
		psc = _psc
		stim_matrix = _stim_matrix
	else:
		psc = _psc[trials]
		stim_matrix = _stim_matrix[:, trials]

	N, K = stim_matrix.shape

	# Load model estimates
	mu, _lam = [model.state[key] for key in ['mu', 'lam']]
	if 'z' in model.state.keys():
		_z = model.state['z']

	if trials is None:
		lam = _lam
		z = _z
	else:
		lam = _lam[:, trials]
		z = _z[trials]

	if ymax is None:
		ymax = np.percentile(psc/np.max(psc), 99.99)
	ymin = -0.05 * ymax
	
	mu_normalised = mu/np.max(mu)
	mu_order = np.argsort(mu)[::-1]

	if order is None:
		order = mu_order.copy()

	# Plotting params
	num_trials = max_trials_to_show

	if fig_width is None:
		fig_width = num_trials * col_width

	normalisation_factor = np.max(np.abs(psc))
	trace_linewidth = 1.5

	I = np.array([np.unique(stim_matrix[:, k])[1] for k in range(stim_matrix.shape[1])])
	powers = np.unique(I)
	trials_per_power = num_trials // len(powers)
	
	# Setup fig
	fig = plt.figure(figsize=(fig_width, row_height * n_plots * 1.5))
	nrows = n_plots + 2 if append_last_row else n_plots
	gs = fig.add_gridspec(ncols=1, nrows=nrows, hspace=hspace, wspace=wspace)

	# Plot checkerboard rows
	for m in range(n_plots):
		n = order[m]

		# spike predictions
		ax = fig.add_subplot(gs[m])
		if m == 0:
			plt.title('Power', fontsize=fontsize, y=1.5)

		stim_locs = np.array([])
		for pwr in powers:
			stim_locs = np.concatenate([stim_locs, np.where(stim_matrix[n] == pwr)[0][:trials_per_power]])

		stim_locs = stim_locs.astype(int)
		this_y_psc = psc[stim_locs].flatten()/normalisation_factor
		n_repeats = np.min([len(stim_locs), max_trials_to_show])
		trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)
		plt.xlim([0, trial_len*n_repeats])

		# Colours
		trace_col = 'k' if mu[n] != 0 else 'gray'
		if facecol is None:
			facecol = 'lightcoral'
		if edgecol is None:
			edgecol = 'None'
			
		for tb in range(len(trial_breaks) - 1):
			if tb > 0:
				plt.plot([trial_breaks[tb], trial_breaks[tb]], [ymin, ymax], '--', color=trace_col)
				
			if not np.isnan(mu_normalised[n]):
				ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=facecol, 
								 edgecolor=edgecol, alpha=lam[n, stim_locs][tb], zorder=-5, linewidth=1.25)
			
			# Plot power changes
			if (m == 0) and (I[stim_locs][tb] != I[stim_locs][tb-1]):
				plt.text(trial_breaks[tb], 1.1 * ymax, '%i mW'%I[stim_locs][tb], fontsize=fontsize-2)
				
			# Annotate estimated sdev
			if sdevs is not None:
				plt.text(trial_breaks[tb] + trial_len/3, 0.65, '%.2f'%sdevs[stim_locs][tb], fontsize=5)
				
			if annotate_spikes:
				# Annotate estimated spiking cells
				assert true_spikes is not None

				inferred_spiking_cells = np.where(lam[:, stim_locs][:, tb] >= 0.5)[0]
				annotated_cells = np.intersect1d(inferred_spiking_cells, np.where(mu != 0)[0])
				annotated_locs = np.array([np.where(order == i)[0][0] + 1 for i in annotated_cells])
				num_spiking_inferred = len(annotated_locs)

				true_spiking_cells = np.where(true_spikes[:, stim_locs][:, tb] > 0)[0]
				annotated_cells = np.intersect1d(true_spiking_cells, np.where(true_weights != 0)[0])
				annotated_locs = np.array([np.where(order == i)[0][0] + 1 for i in annotated_cells])
				num_spiking_true = len(annotated_locs)

				if num_spiking_inferred > 0:
					plt.text(trial_breaks[tb] + trial_len//4, -0.3, '%i: %i'%(num_spiking_true, num_spiking_inferred), fontsize=7)

			if plot_sponts:
				if z[stim_locs][tb] != 0:
					# plt.plot(trial_len * (tb + 0.5), 0.7 * ymax, marker='*', markerfacecolor='b', markeredgecolor='None', markersize=6)
					ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=spont_col, 
								 edgecolor=edgecol, alpha=spont_alpha, zorder=-5, linewidth=1.25)

		plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)
		
		# Overlay true spikes
		if overlay_spikes:
			spk_times = np.array([trial_breaks[tb] for tb in range(len(trial_breaks)-1) if true_spikes[n][stim_locs][tb] == 1])
			plt.scatter(spk_times + trial_len//2, 0.75 * ymax * np.ones_like(spk_times), 20, marker='v', edgecolor='k', facecolor='None', linewidth=0.5)

		for loc in ['top', 'right', 'left', 'bottom']:
			plt.gca().spines[loc].set_visible(False)
		plt.xticks([])
		plt.yticks([])
		plt.ylim([ymin, ymax])
		if labels is not None:
			plt.ylabel(labels[m] + 1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color='k')
		else:
			if m % 4 == 0:
				label_col = 'k' if mu[n] != 0 else 'gray'
				plt.ylabel(m+1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color=label_col)
		
		ax.set_rasterization_zorder(-2)
	
	if append_last_row:
		fig.add_subplot(gs[n_plots])
		plt.axis('off')

		ax = fig.add_subplot(gs[n_plots + 1])
		n = N-1
		stim_locs = np.array([])  
		for pwr in powers:
			stim_locs = np.concatenate([stim_locs, np.where(stim_matrix[n] == pwr)[0][:trials_per_power]])

		stim_locs = stim_locs.astype(int)
		this_y_psc = psc[stim_locs].flatten()/normalisation_factor
		n_repeats = np.min([len(stim_locs), max_trials_to_show])
		trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)

		plt.xlim([0, trial_len*n_repeats])
		trace_col = 'gray'
		for tb in range(len(trial_breaks) - 1):
			if tb > 0:
				plt.plot([trial_breaks[tb], trial_breaks[tb]], [ymin, ymax], '--', color=trace_col)

		plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)
		plt.xticks([])
		plt.yticks([])
		plt.ylim([ymin, ymax])
		for loc in ['top', 'right', 'left', 'bottom']:
			plt.gca().spines[loc].set_visible(False)
		plt.xlabel('Trials', fontsize=fontsize)
		plt.ylabel(N, fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color='gray')
	
	fig.supylabel('Neuron', fontsize=fontsize, x=ylabelpad)
	if save is not None:
		plt.savefig(save, format=save_fmt, bbox_inches='tight', dpi=300, backend=backend)

	plt.show()

def get_cell_order(weights):
    N = weights[0].shape[0]
    cell_order = np.array([])
    for wgt in weights:
        cnx = np.sort(np.where(wgt)[0])[::-1]
        cell_order = np.concatenate([cell_order, np.setdiff1d(cnx, cell_order)])
    cell_order = np.concatenate([cell_order, np.setdiff1d(np.arange(N), cell_order)])
    return cell_order.astype(int)

def plot_spike_inference_comparison(den_pscs, stim_matrices, models, spks=None, titles=None, save=None, ymax=1.1, n_plots=15, max_trials_to_show=30, 
	col_widths=None, row_height=0.6, order=None, trial_len=900, lp_cell=None, fontsize=14):
	# assumes each model in models is the state dictionary.

	if col_widths is None:
		col_widths = 7.5 * np.ones(len(models))
		
	N = stim_matrices[0].shape[0]
	Is = [np.array([np.unique(stim[:, k])[1] for k in range(stim.shape[1])])
		 for stim in stim_matrices]
	ncols = len(models)
	
	fig = plt.figure(figsize=(np.sum(col_widths), row_height * n_plots * 1.5))
	gs = fig.add_gridspec(ncols=ncols, nrows=n_plots, hspace=0.5, wspace=0.05, width_ratios=col_widths/col_widths[0])
	
	normalisation_factor = np.max(np.abs(np.vstack(den_pscs)))
	mu_norm = np.max(np.abs([model['mu'] for model in models]))
	ymin = -0.05 * ymax
	
	trace_linewidth = 0.65
	
	if order is None:
		order = get_cell_order([model['mu'] for model in models])
	
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
			facecol = 'firebrick' if n != lp_cell else 'C0'
			model = models[col]
			lam = model['lam']
			K = lam.shape[1]
			mu = model['mu'].copy()
			trace_col = 'k' if mu[n] != 0 else 'gray'
			
			if 'z' in model.keys():
				z = model['z']
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
				plt.ylabel('%i (%i)'(m+1, n), fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color=label_col)

			ax.set_rasterization_zorder(-2)
	
	if save is not None:
		plt.savefig(save, format='png', bbox_inches='tight', dpi=300, facecolor='white')
	plt.show()