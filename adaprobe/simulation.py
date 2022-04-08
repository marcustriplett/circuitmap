import numpy as np
import itertools
from jax import jit, vmap, partial
import jax.numpy as jnp
from jax.ops import index_update
from jax.nn import sigmoid

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

def simulate(N=300, T=900, H=10, trials=1000, nreps=10, connection_prob=0.05, powers=[45, 55, 65], min_latency=160, gamma_beta=1.5e1, sigma=6e-4,
			frac_strongly_connected=0.2, strong_weight_lower=20, strong_weight_upper=40, weak_exp_mean=4, min_weight=5, phi_0_lower=0.2, phi_0_upper=0.25,
			phi_1_lower=10, phi_1_upper=15, mult_noise_log_var=0.01, tau_r_min=25, tau_r_max=60, tau_delta_min=75, 
			tau_delta_max=250, weights=None, kernel=None, phi_0=None, phi_1=None, gp_scale=4e-3, gp_lengthscale=50, spont_prob=0.05,
			design='random', max_power_min_spike_rate=0.):
	
	assert design in ['random', 'blockwise']

	print('Creating simulation with specifications:')
	print('N', N)
	print('T', T)
	print('H', H)
	print('Trials', trials)
	print('Hologram repetitions', nreps)
	print('Connection density', connection_prob)
	print('Spontaneous PSC probability', spont_prob)
	print('Powers', powers)

	Trange = np.arange(T)
	
	if design == 'blockwise':

		stim_matrix = []
		K = 0
		powers = np.sort(powers)[::-1] # prioritise filling in higher powers
		while K < trials:
			neuron_order = np.random.choice(N, N, replace=False)
			holos = [neuron_order[i*H: (i+1)*H] for i in range(int(np.ceil(N/H)))] # H-spot holograms
			for (power, holo, rep) in itertools.product(powers, holos, range(nreps)): # loop through repetitions first
				if K >= trials:
					break
				stim_trial = np.zeros(N)
				stim_trial[holo] = power
				stim_matrix += [stim_trial]
				K += 1

		reorder = np.random.choice(K, K, replace=False) # shuffle trials
		stim_matrix = np.array(stim_matrix).T
		stim_matrix = stim_matrix[:, reorder]

	if design == 'random':
		# variable `nreps` not used along this path
		K = trials
		stim_matrix = np.zeros((N, K))

		power_order = np.random.choice(np.concatenate(np.array([p * arr for p, arr in zip(powers, np.split(np.ones(K), len(powers)))])), \
			K, replace=False) # equal power representation

		for k in range(K):
			tars = np.random.choice(N, H)
			power = power_order[k]
			stim_matrix[tars, k] = power

	I = np.array([np.unique(stim_matrix[:, k])[-1] for k in range(K)])
	
	# Sample kernel parameters
	if kernel is None:
		tau_r = np.random.uniform(tau_r_min, tau_r_max, N)
		tau_delta = np.random.uniform(tau_delta_min, tau_delta_max, N)
		tau_d = tau_r + tau_delta
		kernel = get_kernels(tau_r, tau_d)
	
	# Biophysical parameters
	if phi_0 is None or phi_1 is None:
		phi_0 = np.random.uniform(phi_0_lower, phi_0_upper, N)
		phi_1 = np.random.uniform(phi_1_lower, phi_1_upper, N)
	sigmoid = lambda x: 1/(1 + np.exp(-x))
	frates = np.array([sigmoid(phi_0 * stim_matrix[:, k] - phi_1) for k in range(K)]).T * (stim_matrix > 0)
	spks = (np.random.rand(N, K) <= frates).astype(float)
	noise = np.random.normal(0, sigma, [K, T])
	mult_noise = np.random.lognormal(0, mult_noise_log_var, [N, K])

	# Pad spikes to ensure min spike rate at maximal power is met
	max_power = np.max(powers)
	for n in range(N):
		locs = np.where(stim_matrix[n] == max_power)[0]
		fr = np.mean(spks[n, locs])
		fr_diff = max_power_min_spike_rate - fr
		if fr_diff > 0:
			# condition not met
			zero_locs = np.where(spks[n, locs] == 0)[0]
			req_spks = int(np.ceil(fr_diff * locs.shape[0]))
			spks[n, locs[np.random.choice(zero_locs, req_spks, replace=False)]] = 1.
	
	spk_times = np.zeros((N, K))
	for n in range(N):
		for k in range(K):
			if spks[n, k]:
				spk_times[n, k] = sample_spike_time(stim_matrix[n, k], gamma_beta=gamma_beta)
	
	if weights is None:
		connected = np.random.rand(N) <= connection_prob
		n_connected = len(np.where(connected)[0])
		n_strongly_connected = int(np.ceil(frac_strongly_connected * n_connected))
		strongly_connected = np.random.choice(np.where(connected)[0], n_strongly_connected, replace=False)
		n_weakly_connected = n_connected - n_strongly_connected
		weakly_connected = np.setdiff1d(np.where(connected)[0], strongly_connected)

		weights = np.zeros(N)
		weights[strongly_connected] = np.random.uniform(strong_weight_lower, strong_weight_upper, n_strongly_connected)
		weights[weakly_connected] = np.random.exponential(weak_exp_mean, n_weakly_connected) + min_weight
		
	else:
		connected = weights != 0
	
	# Generate traces
	print('Generating PSC traces...')
	population_pscs = np.zeros((N, K, T))
	spont_pscs = []
	for n in tqdm(range(N), desc='Neuron', leave=True):
		kern = kernel[n](Trange[:, None], spk_times[n]).T
		population_pscs[n] = (mult_noise[n] * spks[n])[:, None] * weights[n] * kern/np.trapz(kern, axis=-1)[:, None]

	print('Generating spontaneous PSCs...')
	spont_pscs = np.zeros((K, T))
	for k in range(K):
		if np.random.rand() <= spont_prob:
			# spontaneous event
			tau_r_sample = np.random.uniform(tau_r_min, tau_r_max)
			tau_delta_sample = np.random.uniform(tau_delta_min, tau_delta_max)
			tau_d_sample = tau_r_sample + tau_delta_sample
			spont_kern = get_kernel(tau_r_sample, tau_d_sample)
			spike_time_sample = np.random.randint(10, 500)
			weight_sample = np.random.uniform(np.min(weights[connected]), np.max(weights[connected]))
			kern = spont_kern(Trange, spike_time_sample)
			spont_psc = weight_sample * kern/np.trapz(kern)
			spont_pscs[k] = spont_psc
	
	# Sample correlated noise
	print('Sampling correlated noise from Gaussian process...')
	gp_noise = sample_gp(trial_dur=T, n_samples=K, gp_scale=gp_scale, gp_lengthscale=gp_lengthscale)

	psc = np.sum(population_pscs, axis=0) + spont_pscs + gp_noise + noise
	
	sim = {
		'weights': weights,
		'phi_0': phi_0,
		'phi_1': phi_1,
		'mult_noise': mult_noise,
		'sigma': sigma,
		'stim_matrix': stim_matrix,
		'psc': psc,
		'gp_noise': gp_noise,
		'kernel': kernel,
		'spks': spks,
		'spk_times': spk_times,
		'spont_pscs': spont_pscs,
		'I': I,
	}
	
	print('Complete.\n')
	return sim

def alpha(power, scale=1e4):
	return scale/(power**2)

def get_kernel(tau_r, tau_d):
	def func(t, delta):
		return (np.exp(-(t - delta)/tau_d) - np.exp(-(t - delta)/tau_r)) * (t > delta)
	return func

def get_kernels(tau_r, tau_d):
	return [get_kernel(tr, td) for (tr, td) in zip(tau_r, tau_d)]

def sample_spike_time(power, gamma_beta=1.5e1, min_latency=160):
	return min_latency + np.random.gamma(alpha(power), gamma_beta)

def sample_gp(trial_dur=900, gp_lengthscale=25, gp_scale=0.01, n_samples=1):
	D = np.array([[i - j for i in range(trial_dur)] for j in range(trial_dur)])
	K = np.exp(-D**2/(2 * gp_lengthscale**2))
	mean = np.zeros(trial_dur)
	return gp_scale * np.random.multivariate_normal(mean, K, size=n_samples)

def _eval_kernel(trange, tau_r, tau_d, delta, eps=1e-8):
	ke = (jnp.exp(-(trange - delta)/tau_d) - jnp.exp(-(trange - delta)/tau_r)) * (trange > delta)
	return ke/(jnp.max(ke) + eps)
eval_kernel = jit(vmap(_eval_kernel, in_axes=(None, 0, 0, 0)))

def simulate_continuous_experiment_without_spike_failures(N=100, connected_frac=0.2, exp_len=int(2e4), gamma_beta=1.5e1, min_latency=60, spont_rate=0.0005, 
	mult_noise_log_var=0.01, response_length=900, noise_std=1e-2, tau_r_min=10, tau_r_max=40, tau_delta_min=250,
	tau_delta_max=300, power=50, sampling_freq=20000, stim_freq=10, weight_lower=2, weight_upper=10, seed=0, ar_coef=0.95, ar_std=1e-1):
	'''Simulate continuous mapping experiment (to be sliced and reshaped post-hoc).
	'''
	# time constants
	tau_r = np.random.uniform(tau_r_min, tau_r_max, N)
	tau_delta = np.random.uniform(tau_delta_min, tau_delta_max, N)
	tau_d = tau_r + tau_delta
	trange = np.arange(exp_len)

	# stimulus timing
	isi = sampling_freq/stim_freq
	stim_times = np.arange(isi, exp_len - response_length, isi, dtype=int)
	nstim = len(stim_times)
	spike_times = sample_spike_time(power * np.ones(nstim), gamma_beta=gamma_beta, min_latency=min_latency)
	tars = np.random.choice(N, nstim)

	# connections
	n_connected = int(connected_frac * N)
	connected = np.random.choice(np.arange(N), n_connected, replace=False)
	weights = np.zeros(N)
	weights[connected] = np.random.uniform(weight_lower, weight_upper, n_connected)

	# responses
	mult_noise = np.random.lognormal(0, mult_noise_log_var, [nstim, 1])
	kevals = eval_kernel(trange, tau_r[tars], tau_d[tars], stim_times + spike_times)
	pscs = kevals * weights[tars, None] * mult_noise
	
	# extract ground truth responses
	true_resps = np.array([pscs[n, st-100: st+800] for n, st in enumerate(stim_times)])

	# add spontaneous activity
	nspont = int(spont_rate * exp_len)
	spont_times = np.random.choice(exp_len, nspont, replace=False)
	spont_pscs = np.zeros(exp_len)
	tau_r = np.random.uniform(tau_r_min, tau_r_max, nspont)
	tau_delta = np.random.uniform(tau_delta_min, tau_delta_max, nspont)
	tau_d = tau_r + tau_delta
	sponts = eval_kernel(trange, tau_r, tau_d, spont_times) * \
	 np.random.uniform(weight_lower, weight_upper, [nspont, 1])

	# compute correlated noise
	ar1_noise = np.zeros(exp_len)
	iid_noise = np.random.normal(0, ar_std, exp_len)
	for t in range(1, exp_len):
		ar1_noise[t] = ar_coef * ar1_noise[t-1] + iid_noise[t]
	
	pscs = np.sum(pscs, axis=0) + np.sum(sponts, axis=0) + ar1_noise
	obs_resps = np.array([pscs[st-100: st+800] for n, st in enumerate(stim_times)])

	expt = {
		'pscs': pscs,
		'obs_responses': obs_resps,
		'true_responses': true_resps,
		'tars': tars,
		'stim_times': stim_times
	}
	
	return expt

#% simulate_continuous_experiment helper funcs
def _get_psc_kernel(tau_r, tau_d, kernel_window, eps=1e-5):
	krange = jnp.arange(kernel_window)
	ke = jnp.exp(-krange/tau_d) - jnp.exp(-krange/tau_r) # normalised kernel
	return ke/(jnp.trapz(ke) + eps)
get_psc_kernel = jit(vmap(_get_psc_kernel, in_axes=(0, 0, None)), static_argnums=(2))

def _get_unnormalised_psc_kernel(tau_r, tau_d, kernel_window, eps=1e-5):
	krange = jnp.arange(kernel_window)
	ke = jnp.exp(-krange/tau_d) - jnp.exp(-krange/tau_r) # normalised kernel
	return ke
get_unnormalised_psc_kernel = jit(vmap(_get_unnormalised_psc_kernel, in_axes=(0, 0, None)), static_argnums=(2))

def _kernel_conv(trange, psc_kernel, delta, spike, mult_noise, weight):
	''' Warning: assumes no spike occurs on very first bin due to jax workaround.
	'''
	stimv = jnp.zeros(trange.shape[0])
	locs = (delta * spike).astype(int)
	stimv = index_update(stimv, locs, weight * mult_noise)
	stimv = index_update(stimv, 0, 0)
	return jnp.convolve(psc_kernel, stimv, mode='full')[:trange.shape[0]]

_vmap_kernel_conv = vmap(_kernel_conv, in_axes=(None, 0, 0, 0, 0, 0))

@jit
def kernel_conv(trange, psc_kernel, delta, spike, mult_noise, weight):
	''' Compute PSCs via convolution of weighted spikes with PSC kernel.
	'''
	return jnp.sum(_vmap_kernel_conv(trange, psc_kernel, delta, spike, mult_noise, weight), axis=0)

def _eval_sponts(trange, tau_r, tau_d, delta, weight, divisor, eps=1e-8):
	ke = jnp.nan_to_num((jnp.exp(-(trange - delta)/tau_d) - jnp.exp(-(trange - delta)/tau_r)) * (trange > delta))
	return (ke * weight)/(divisor + eps)

eval_sponts = jit(lambda *args: jnp.sum(vmap(_eval_sponts, in_axes=(None, 0, 0, 0, 0, 0))(*args), axis=0))

def _get_true_evoked_resp(spike_time, noise_weighted_spike, weight, psc_kernel, response_length=900, prior_context=100):
	stimv = index_update(jnp.zeros(response_length), (prior_context + spike_time).astype(int), noise_weighted_spike * weight)
	return jnp.convolve(stimv, psc_kernel)[:stimv.shape[0]]

get_true_evoked_resp = lambda *args: jnp.sum(vmap(_get_true_evoked_resp, in_axes=(0, 0, 0, 0))(*args), axis=0)
get_true_evoked_resp_vmap = jit(vmap(get_true_evoked_resp, in_axes=(1, 1, None, None)))

def simulate_continuous_experiment(N=100, expt_len=int(2e4), gamma_beta=1.5e1, min_latency=60, powers=[45, 55, 65],
	mult_noise_log_var=0.05, response_length=900, noise_std=1e-2, tau_r_min=10, tau_r_max=40, tau_delta_min=250, tau_delta_max=300, sampling_freq=20000, 
	stim_freq=10, weight_lower=2, weight_upper=10, seed=0, ar_coef=0.95, ar_std=1e-1, weights=None, frac_strongly_connected=0.2, strong_weight_lower=20, 
	strong_weight_upper=40, weak_exp_mean=4, min_weight=5, phi_0_lower=0.2, phi_0_upper=0.25, phi_1_lower=10, phi_1_upper=15, kernel=None, phi_0=None, phi_1=None,
	H=10, nreps=1, connection_prob=0.1, spont_rate=10, kernel_window=3000, prior_context=100, ground_truth_eval_batch_size=1000):
	'''Simulate continuous mapping experiment (to be sliced and reshaped post-hoc).
	'''
	
	print('Creating simulation with specifications:')
	print('Population size', N)
	print('Hologram targets', H)
	print('Response window (frames)', response_length)
	print('Stimulus frequency (Hz)', stim_freq)
	print('Experiment duration (s) %.2f'%(expt_len/sampling_freq))
	print('Sampling frequency (KHz)', sampling_freq/1000)
	print('Hologram repetitions', nreps)
	print('Connection density', connection_prob)
	print('Spontaneous PSC rate (Hz)', spont_rate)
	print('Powers', powers)
	
	# time constants
	tau_r = np.random.uniform(tau_r_min, tau_r_max, N)
	tau_delta = np.random.uniform(tau_delta_min, tau_delta_max, N)
	tau_d = tau_r + tau_delta
	trange = np.arange(expt_len)

	# synaptic weights
	if weights is None:
		n_connected = int(connection_prob * N)
		connected = np.random.choice(np.arange(N), n_connected, replace=False)
		n_strongly_connected = int(np.ceil(frac_strongly_connected * n_connected))
		strongly_connected = np.random.choice(connected, n_strongly_connected, replace=False)
		weakly_connected = np.setdiff1d(connected, strongly_connected)
		n_weakly_connected = len(weakly_connected)

		weights = np.zeros(N)
		weights[strongly_connected] = np.random.uniform(strong_weight_lower, strong_weight_upper, n_strongly_connected)
		weights[weakly_connected] = np.random.exponential(weak_exp_mean, n_weakly_connected) + min_weight

	else:
		connected = np.where(weights != 0)[0]
	
	# stimulus timing
	isi = sampling_freq/stim_freq
	stim_times = np.arange(isi, expt_len - response_length - isi, isi, dtype=int)
	nstim = len(stim_times)
	
	# stim design
	spike_times = np.zeros((N, nstim))
	stim_matrix = []
	K = 0
	powers = np.sort(powers)[::-1] # prioritise filling in higher powers
	while K < nstim:
		neuron_order = np.random.choice(N, N, replace=False)
		holos = [neuron_order[i*H: (i+1)*H] for i in range(int(np.ceil(N/H)))] # H-spot holograms
		for (power, holo, rep) in itertools.product(powers, holos, range(nreps)): # loop through repetitions first
			if K >= nstim:
				break
			stim_trial = np.zeros(N)
			stim_trial[holo] = power
			stim_matrix += [stim_trial]
			spike_times[holo, K] = sample_spike_time(power * np.ones(holo.shape[0]), gamma_beta=gamma_beta, min_latency=min_latency)
			K += 1        
	stim_matrix = np.array(stim_matrix).T
	reorder = np.random.choice(K, K, replace=False) # shuffle trials
	stim_matrix = stim_matrix[:, reorder]
	spike_times = spike_times[:, reorder]
	
	# responses
	mult_noise = np.random.lognormal(0, mult_noise_log_var, [N, nstim])
	
	if phi_0 is None or phi_1 is None:
		phi_0 = np.random.uniform(phi_0_lower, phi_0_upper, N)
		phi_1 = np.random.uniform(phi_1_lower, phi_1_upper, N)

	frates = np.array([sigmoid(phi_0 * stim_matrix[:, k] - phi_1) for k in range(K)]).T * (stim_matrix > 0)
	spks = (np.random.rand(N, K) <= frates).astype(float)
	
	# compute pscs
	psc_kernels = get_psc_kernel(tau_r, tau_d, kernel_window)
	pscs = np.array(kernel_conv(trange, psc_kernels[connected], spike_times[connected] + stim_times[np.newaxis], 
					   spks[connected], mult_noise[connected], weights[connected]))
	
	# extract ground truth responses
	#% batch-wise for controlling memory overhead
	true_resps = []
	nbatches = int(np.ceil(nstim/ground_truth_eval_batch_size))
	for i in range(nbatches):
		true_resps += [get_true_evoked_resp_vmap(
			spike_times[:, i*ground_truth_eval_batch_size:(i+1)*ground_truth_eval_batch_size],
			(spks * mult_noise)[:, i*ground_truth_eval_batch_size:(i+1)*ground_truth_eval_batch_size],
			weights,
			psc_kernels
		)]
	true_resps = np.array(jnp.concatenate(true_resps))
	
	# add spontaneous activity
	nspont = int(spont_rate/sampling_freq * expt_len)
	spont_times = np.random.choice(expt_len, nspont, replace=False)
	tau_r = np.random.uniform(tau_r_min, tau_r_max, nspont)
	tau_delta = np.random.uniform(tau_delta_min, tau_delta_max, nspont)
	tau_d = tau_r + tau_delta
	psc_kernels = get_unnormalised_psc_kernel(tau_r, tau_d, kernel_window)
	kernel_divisor = np.trapz(psc_kernels, axis=-1)

	sponts = np.array(eval_sponts(trange, tau_r, tau_d, spont_times, np.random.uniform(min_weight, np.max(weights), [nspont]), kernel_divisor))

	# compute correlated noise
	ar1_noise = np.zeros(expt_len)
	iid_noise = np.random.normal(0, ar_std, expt_len)
	ar1_noise[0] = iid_noise[0]
	for t in range(1, expt_len):
		ar1_noise[t] = ar_coef * ar1_noise[t-1] + iid_noise[t]

	pscs = pscs + sponts + ar1_noise
	obs_resps = np.array([pscs[st-prior_context: st+response_length-prior_context] for st in stim_times])
	
	expt = {
		'pscs': pscs,
		'obs_responses': obs_resps,
		'true_responses': true_resps,
		'stim_matrix': stim_matrix,
		'weights': weights
	}

	return expt

