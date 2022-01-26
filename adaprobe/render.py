import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def render():

	# Set-up initial frame
	xrange = np.arange(-20, 20, 0.1)
	contours = [None] * 3 # contourf objects
	top_axes = [None] * 3 # top axes containing contourf objects
	cb_axes = [None] * 3 # colorbar axes
	_power = powers[guiding_power]
	rf_plot_scale = 70

	fig = plt.figure(figsize=(17, 9.5))
	gs = fig.add_gridspec(4, 6, height_ratios=[1, 0.25, 0.25, 0.25], width_ratios=[1, 0.05, 1, 0.05, 1, 0.05])

	for i in range(3):
		top_axes[i] = fig.add_subplot(gs[0, 2*i])
		contours[i] = top_axes[i].contourf(gridx, gridy, Z_collection[0, i].reshape(len(yr), len(xr)), 150)
		cb_axes[i] = fig.add_subplot(gs[0, 2*i + 1])
		plt.colorbar(contours[0], cax=cb_axes[i], format='%1.3f')
		adjust_ax(cb_axes[i])

	ax4 = fig.add_subplot(gs[1, :])
	ax4.bar(np.arange(N), alpha_hist[0])
	ax4.set_ylim([0, 1])
	ax4.set_xticks(range(N))

	ax5 = fig.add_subplot(gs[2, :])
	for n in range(N):
		ax5.plot(xrange/50 + n, gauss(mu_hist[0, n], beta_hist[0, n], xrange), color='k')
	ax5.set_xticks(range(N))
	ax5.set_xlabel('Neuron')

	## Receptive field plots
	xrange_rf = np.arange(0, 50, 0.01)
	rf_ax = fig.add_subplot(gs[3, :])
	for n in range(N):
		fn = sigmoid(phi_hist[0, n, 0] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - phi_hist[0, n, 1])
		fn_lower = sigmoid((phi_hist[0, n, 0] - np.sqrt(phi_cov_hist[0, n, 0, 0])) * _power * np.exp(-model.state['omega'][n] * xrange_rf) - (phi_hist[0, n, 1] - np.sqrt(phi_cov_hist[0, n, 1, 1])))
		fn_upper = sigmoid((phi_hist[0, n, 0] + np.sqrt(phi_cov_hist[0, n, 0, 0])) * _power * np.exp(-model.state['omega'][n] * xrange_rf) - (phi_hist[0, n, 1] + np.sqrt(phi_cov_hist[0, n, 1, 1])))
		
		if sim.w[n] == 0:
			rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, sigmoid(sim.phi_0[n] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - sim.phi_1[n]), ':', color='gray')
		else:
			rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, sigmoid(sim.phi_0[n] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - sim.phi_1[n]), '--', color='k')
		rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, fn, color='k')
		rf_ax.fill_between((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, fn_lower, fn_upper, facecolor='firebrick', edgecolor='None', alpha=0.25)
		
	rf_ax.set_xticks(range(N))
	rf_ax.set_xticklabels(range(1, N + 1))

	plot_scale = 50
	def animate(t):
		print('animating frame %i/%i'%(t+1, ntrials), end='\r')
		for ax in top_axes + [ax4, ax5, rf_ax]:
			ax.clear()
		for ax in cb_axes:
			ax.cla()
		for indx, ax in enumerate(top_axes):
			if indx == guiding_power:
				ax.set_title('%i mW (guide)'%powers[indx])
			else:
				ax.set_title('%i mW'%powers[indx])
			contours[indx] = ax.contourf(gridx, gridy, Z_collection[t, indx].reshape(len(yr), len(xr)), 150) # replot contours
			ax.scatter(model.cell_locs[:, 0], model.cell_locs[:, 1], marker='x', color='white')

		for i in range(3):
			plt.colorbar(contours[i], cax=cb_axes[i], format='%1.3f')
		
		ax4.bar(np.arange(N), alpha_hist[t], color='firebrick')
		ax4.set_xticks(range(N))
		ax4.set_xticklabels(range(1, N + 1))
		ax4.set_ylim([0, 1])
		ax4.set_ylabel('Connection prob', fontsize=fontsize-2)

		gauss_vec = np.array([gauss(mu_hist[t, n], beta_hist[t, n], xrange) for n in range(N)])
		for n in range(N):
			ax5.plot(xrange/plot_scale + n, gauss_vec[n], color='firebrick')
			if sim.w[n] == 0:
				ax5.plot([sim.w[n]/plot_scale + n, sim.w[n]/plot_scale + n], [0, 0.9 * np.max(gauss_vec)], ':', color='gray')
			else:
				ax5.plot([sim.w[n]/plot_scale + n, sim.w[n]/plot_scale + n], [0, 0.9 * np.max(gauss_vec)], '--', color='k')
		ax5.set_xticks(range(N))
		ax5.set_xticklabels(range(1, N + 1))
		ax5.set_xlabel('Neuron', fontsize=fontsize-2)
		ax5.set_ylabel('Density', fontsize=fontsize-2)

		for n in range(N):
			# Calculate posterior error bars
			fn = sigmoid(phi_hist[t, n, 0] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - phi_hist[t, n, 1])
			fn_lower = sigmoid((phi_hist[t, n, 0] - np.sqrt(phi_cov_hist[t, n, 0, 0])) * _power * np.exp(-model.state['omega'][n] * xrange_rf) - (phi_hist[t, n, 1] - np.sqrt(phi_cov_hist[t, n, 1, 1])))
			fn_upper = sigmoid((phi_hist[t, n, 0] + np.sqrt(phi_cov_hist[t, n, 0, 0])) * _power * np.exp(-model.state['omega'][n] * xrange_rf) - (phi_hist[t, n, 1] + np.sqrt(phi_cov_hist[t, n, 1, 1])))
			
			if sim.w[n] == 0:
				rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, sigmoid(sim.phi_0[n] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - sim.phi_1[n]), ':', color='gray')
			else:
				rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, sigmoid(sim.phi_0[n] * _power * np.exp(-model.state['omega'][n] * xrange_rf) - sim.phi_1[n]), '--', color='k')
			rf_ax.plot((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, fn, color='k')
			rf_ax.fill_between((xrange_rf - np.max(xrange_rf)/2)/rf_plot_scale + n, fn_lower, fn_upper, facecolor='firebrick', edgecolor='None', alpha=0.25)
		rf_ax.set_xticks(range(N))
		rf_ax.set_ylabel('Spike prob\n(%i mW)'%powers[guiding_power], fontsize=fontsize-2)
		rf_ax.set_xlabel('Neuron', fontsize=fontsize-2)
		rf_ax.set_xticklabels(range(1, N + 1))
		
		for ax in top_axes:
			for n in range(N):
				ax.text(model.cell_locs[n, 0], model.cell_locs[n, 1], n + 1, fontsize=fontsize)
			if rand_hist[t] == 1:
				ax.scatter(sim.L[t][0], sim.L[t][1], 500, marker='o', facecolor='None', edgecolor='cyan')
			else:
				ax.scatter(sim.L[t][0], sim.L[t][1], 500, marker='o', facecolor='None', edgecolor='red')
		
		top_axes[-1].text(34, 41, 'trial %i'%t, fontsize=fontsize, color='white')

	anim = matplotlib.animation.FuncAnimation(fig, animate, frames=range(600))