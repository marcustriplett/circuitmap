#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import adaprobe
from adaprobe.psc_denoiser import NeuralDenoiser
from adaprobe.simulation import simulate
import pickle
import glob
import neursim.utils as util
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import torch
import gc



def denoise_pscs_in_batches(psc, denoiser, batch_size=4096):

    num_pscs = psc.shape[0]
    num_batches = np.ceil(num_pscs / batch_size)
    den_psc_batched = [denoiser(batch, verbose=False) for batch in np.array_split(psc, num_batches, axis=0)]
    return np.concatenate(den_psc_batched)


# Make stim matrix for mbcs.
# Matrix dimensions are num_trials x num_neurons
# where element ij is the power used to stimulate neuron j
# on trial i
def make_stim_matrix(psc, I, L):
    grid = util.make_grid_from_stim_locs(L)

    # make map which takes location and maps to index
    loc_map = {tuple(loc) : idx for idx, loc in enumerate(grid.flat_grid)}

    num_neurons, num_trials = len(loc_map), psc.shape[0]
    stim = np.zeros((num_neurons, num_trials))
    neuron_idxs = np.array([loc_map[tuple(loc)] for loc in L])
    stim[neuron_idxs, np.arange(num_trials)] = I

    return stim

def separate_data_by_plane(psc, I, L):
    grid = util.make_grid_from_stim_locs(L)

    stim_mats = []
    pscs = []
    Is = []
    Ls = []
    for z_idx, z in enumerate(grid.zs):

        # make map which takes location and maps to index
        # for a single plane only
#         import pdb; pdb.set_trace()
        this_plane = grid.flat_grid[:,-1] == z
        loc_map = {tuple(loc) : idx for idx, loc in enumerate(grid.flat_grid[this_plane])}

        # get number of neurons (i.e pixels) for a single plane
        # and number of trials in that plane.
        these_trials = L[:,-1] == z
        num_neurons, num_trials = len(loc_map), sum(these_trials)
        stim = np.zeros((num_neurons, num_trials))

        # convert from L -> neuron_idx (number between 0 and num_neurons - 1)
        neuron_idxs = np.array([loc_map[tuple(loc)] for loc in L[these_trials]])

        # set entries of stim matrix. Since this is single spot data,
        # we can set the entire matrix at once (this is
        # trickier with multispot data)
        stim[neuron_idxs, np.arange(num_trials)] = I[these_trials]

        # save stim and corresponding traces
        stim_mats.append(stim)
        pscs.append(psc[these_trials])
        Is.append(I[these_trials])
        Ls.append(L[these_trials])
    return stim_mats, pscs, Is, Ls



def make_priors_mbcs(N,K):
    beta_prior = 3e0 * np.ones(N)
    mu_prior = np.zeros(N)
    rate_prior = 1e-1 * np.ones(K)
    shape_prior = np.ones(K)

    priors = {
        'beta': beta_prior,
        'mu': mu_prior,
        'shape': shape_prior,
        'rate': rate_prior,
    }
    return priors

def make_priors_vsns(N,K):
    phi_prior = np.c_[0.125 * np.ones(N), 5 * np.ones(N)]
    phi_cov_prior = np.array([np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])
    alpha_prior = 0.15 * np.ones(N)
    beta_prior = 3e0 * np.ones(N)
    mu_prior = np.zeros(N)
    sigma_prior = np.ones(N)
    sigma=1

    priors_vsns = {
        'alpha': alpha_prior,
        'beta': beta_prior,
        'mu': mu_prior,
        'phi': phi_prior,
        'phi_cov': phi_cov_prior,
        'shape': 1.,
        'rate': sigma**2,
    }

    return priors_vsns


def denoise_grid(model_type, fit_options, den_pscs, stims, trial_keep_prob=0.1):
    all_models = []

    if model_type == 'mbcs':
        prior_fn = make_priors_mbcs
        method = 'mbcs_spike_weighted_var_with_outliers'
    elif model_type == 'variational_sns':
        prior_fn = make_priors_vsns
        method = 'cavi_sns'
    else:
        raise ValueError("invalid model type...")

    for psc, stim in zip(den_pscs, stims):


        # subsample_trials
        num_trials = psc.shape[0]
        keep_idx = np.random.rand(num_trials) <= trial_keep_prob
        psc = psc[keep_idx,...]
        stim = stim[:, keep_idx]

        # create priors and model
        N,K = stim.shape
        priors = prior_fn(N,K)
        model_params = {'N': N, 'model_type': model_type, 'priors': priors}
        model = adaprobe.Model(**model_params)

        # fit model and save
        model.fit(psc, stim, fit_options=fit_options, method=method)
        all_models.append(model)

    return all_models





def plot_multi_means(fig, mean_maps, depth_idxs, zs=None, powers=None, map_names=None):

    for mean_idx, mean_map in enumerate(mean_maps):

        num_powers, _, _, num_planes = mean_map.shape

        num_planes_to_plot = len(depth_idxs)
        assert num_planes_to_plot <= num_planes


        # Create a new grid for each mean map
        subplot_args = int("1" + str(len(mean_maps)) + str(mean_idx + 1))
        ax_curr = plt.subplot(subplot_args)



        if powers is not None and map_names is not None:
            ax_curr.set_title(map_names[mean_idx], y=1.08)

        plt.axis('off')

        grid = ImageGrid(fig, subplot_args,  # similar to subplot(111)
                         nrows_ncols=(num_planes_to_plot, num_powers),  # creates 2x2 grid of axes
                         axes_pad=0.05,  # pad between axes in inch.
                         cbar_mode='single',
                         cbar_pad=0.2
                         )

        min_val = np.min(mean_map)
        max_val = np.max(mean_map)

        for j, ax in enumerate(grid):
            row = j // num_powers
            col = j % num_powers
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            # optionally add labels
            if zs is not None and col == 0:
                ax.set_ylabel('%d ' % zs[depth_idxs[row]] + r'$\mu m $' )

            if powers is not None and row == 0:
                ax.set_title(r'$%d mW$' % powers[col])

            if min_val < 0:
                kwargs = {'cmap':'viridis_r'}

            im = ax.imshow(mean_map[col,:,:,depth_idxs[row]],
                           origin='lower', vmin=min_val, vmax=max_val, **kwargs)

            cbar = grid[0].cax.colorbar(im)

def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='MBCS for Grid Denoising')
    parser.add_argument('--minimax-spike-prob', type=float, default=0.1)
    parser.add_argument('--minimum-spike-count', type=int, default=3)
    parser.add_argument('--num-iters', type=int, default=30)
    parser.add_argument('--model-type', type=str, default='mbcs')
    args = parser.parse_args(argseq)

    # parameters you can vary
    iters = args.num_iters
    minimax_spike_prob = args.minimax_spike_prob
    minimum_spike_count = args.minimum_spike_count

    # fit options for mbcs
    seed = 1
    y_xcorr_thresh = 1e-2
    max_penalty_iters = 50
    warm_start_lasso = True
    verbose = False
    num_mc_samples_noise_model = 100
    noise_scale = 0.5
    init_spike_prior = 0.5
    num_mc_samples = 500
    penalty = 2
    max_lasso_iters = 1000
    scale_factor = 0.75
    constrain_weights = 'positive'
    orthogonal_outliers = True
    lam_mask_fraction = 0.0
    delay_spont_estimation = 2

    fit_options_mbcs = {
        'iters': iters,
        'num_mc_samples': num_mc_samples,
        'penalty': penalty,
        'max_penalty_iters': max_penalty_iters,
        'max_lasso_iters': max_lasso_iters,
        'scale_factor': scale_factor,
        'constrain_weights': constrain_weights,
        'y_xcorr_thresh': y_xcorr_thresh,
        'seed': seed,
        'verbose': verbose,
        'warm_start_lasso': warm_start_lasso,
        'minimum_spike_count': minimum_spike_count,
        'minimum_maximal_spike_prob': minimax_spike_prob,
        'noise_scale': noise_scale,
        'init_spike_prior': init_spike_prior,
        'orthogonal_outliers': orthogonal_outliers,
        'lam_mask_fraction': lam_mask_fraction,
        'delay_spont_estimation': delay_spont_estimation
    }

    # fit options for vsns
    iters = args.num_iters
    minimum_spike_count = args.minimum_spike_count
    minimax_spike_prob = args.minimax_spike_prob
    sigma = 1.
    seed = 1
    y_xcorr_thresh = 1e-2
    max_penalty_iters = 50
    warm_start_lasso = True
    verbose = False
    num_mc_samples_noise_model = 100
    noise_scale = 0.5
    init_spike_prior = 0.5
    num_mc_samples = 500
    penalty = 1e1
    max_lasso_iters = 1000
    scale_factor = 0.75
    constrain_weights = 'positive'
    orthogonal_outliers = True
    lam_mask_fraction = 0.025

    fit_options_vsns = {
        'iters': iters,
        'num_mc_samples': num_mc_samples,
        'y_xcorr_thresh': y_xcorr_thresh,
        'seed': seed,
        'phi_thresh': None,
        'phi_thresh_delay': 5,
        'learn_noise': True,
        'minimax_spk_prob': 0.2,
        'scale_factor': 0.75,
        'penalty': penalty
    }


    if args.model_type == 'mbcs':
        return fit_options_mbcs, args
    elif args.model_type == 'variational_sns':
        return fit_options_vsns, args
    else:
        raise ValueError("Unknown argument for model type")



if __name__ == "__main__":

    fit_options, args = parse_fit_options(sys.argv[1:])

    # load pscs from an example file, then run demixer in batches
    filenames = glob.glob('%s/blind_mapping/blind_maps/*denoised.npz.npy' % os.path.expanduser("~"))

    denoiser = NeuralDenoiser(path='denoisers/seq_unet_50k_ai203_v2.ckpt')

    for file in filenames:
    	data_dict = np.load(file, allow_pickle=True).item()
    	psc = -data_dict['psc']

    	den_psc = denoise_pscs_in_batches(psc, denoiser)	
    	stims, den_pscs, Is, Ls = separate_data_by_plane(den_psc, data_dict['I'], data_dict['L'])
    	plane_models = denoise_grid(args.model_type, fit_options, den_pscs, stims, trial_keep_prob=1.0)


    	mean_raw = util.make_suff_stats(data_dict['psc'].sum(1), data_dict['I'], data_dict['L'])[0]
    	mean_demixed = util.make_suff_stats(-den_psc.sum(1), data_dict['I'], data_dict['L'])[0]

    	# take weights from mbcs models and reshape to mean map
    	# shape will be 1 x 26 x 26 x 5
    	mean_mbcs = np.zeros((1, 26, 26, 5))
    	for i, model in enumerate(plane_models):
    	    mean_mbcs[0,:,:,i] = -np.reshape(model.state['mu'], (26,26)).T


    	depth_idxs = np.arange(5)
    	fig = plt.figure(figsize=(8, 0.75 * 5), dpi=300, facecolor='white')
    	plot_multi_means(fig, [mean_raw, mean_demixed, mean_mbcs], np.arange(5))

    	dataset_name = os.path.basename(file).split('.')[0]
    	plt.savefig(dataset_name + args.model_type + '.png', dpi=300)

    	with open(dataset_name + args.model_type + '_models.pkl', 'wb') as f:
    	    pickle.dump(plane_models, f, pickle.HIGHEST_PROTOCOL)
