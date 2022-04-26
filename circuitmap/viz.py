import matplotlib.pyplot as plt
import numpy as np

def plot_checkerboard(psc, stim_matrix, model, true_spikes=None, true_weights=None, spike_thresh=0.01, save=None, ymax=None, n_plots=15, max_trials_to_show=30, 
						 col_width=7.5, row_height=0.6, order=None, sdevs=None, fig_width=None, overlay_spikes=True, annotate_spikes=False, wspace=0.05, 
						 hspace=0.5, ylabelpad=0.05, facecol=None, edgecol=None, trial_len=900):
	''' plot_checkerboard
	'''

	N, K = stim_matrix.shape
	mu, lam = [model.state[key] for key in ['mu', 'lam']]

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
	
	fig = plt.figure(figsize=(fig_width, row_height * n_plots * 1.5))
	gs = fig.add_gridspec(ncols=1, nrows=n_plots + 2, hspace=hspace, wspace=wspace)
	
	normalisation_factor = np.max(np.abs(psc))
	trace_linewidth = 1.5
	
	I = np.array([np.unique(stim_matrix[:, k])[1] for k in range(stim_matrix.shape[1])])
	powers = np.unique(I)
	trials_per_power = num_trials // len(powers)
	
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

		plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)
		
		# Overlay true spikes
		if overlay_spikes:
			if true_weights[n] != 0:
				spk_times = np.array([trial_breaks[tb] for tb in range(len(trial_breaks)-1) if true_spikes[n][stim_locs][tb] == 1])
				plt.scatter(spk_times + trial_len//2, 0.75 * ymax * np.ones_like(spk_times), 20, marker='v', edgecolor='k', facecolor='None', linewidth=0.5)

		for loc in ['top', 'right', 'left', 'bottom']:
			plt.gca().spines[loc].set_visible(False)
		plt.xticks([])
		plt.yticks([])
		plt.ylim([ymin, ymax])
		if m % 4 == 0:
			label_col = 'k' if true_weights[n] != 0 else 'gray'
			plt.ylabel(m+1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center', color=label_col)
		
		ax.set_rasterization_zorder(-2)
	
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
		plt.savefig(save, format='pdf', bbox_inches='tight', dpi=300, backend='pgf')

	plt.show()