import numpy as np
import itertools

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

def simulate(N=300, T=900, H=10, nreps=10, connection_prob=0.05, powers=[45, 55, 65], min_latency=160, gamma_beta=1.5e1, sigma=6e-4,
			frac_strongly_connected=0.2, strong_weight_lower=20, strong_weight_upper=40, weak_exp_mean=4, min_weight=5, phi_0_lower=0.2, phi_0_upper=0.25,
			phi_1_lower=10, phi_1_upper=15, mult_noise_log_var=0.01, tau_r_min=25, tau_r_max=60, tau_delta_min=75, 
			tau_delta_max=250, weights=None, kernel=None, phi_0=None, phi_1=None, gp_scale=4e-3, gp_lengthscale=50, spont_prob=0.05,
			design='random', trials=None):
	
	assert design in ['random', 'blockwise']

	Trange = np.arange(T)
	
	if H == 1:
		# Design stimulus
		K = nreps * N * len(powers)
		stim_matrix = np.zeros((N, K))

		k = 0
		for n in range(N):
			for p in powers:
				stim_matrix[n, k: k+nreps] = p
				k += nreps
		stim_order = np.random.choice(K, K, replace=False)
		stim_matrix = stim_matrix[:, stim_order]
	else:
		# case H > 1
		if design == 'blockwise':
			assert trials is not None

			stim_matrix = []
			K = 0
			powers = np.sort(powers)[::-1] # prioritise filling in higher powers
			while K < trials:
				neuron_order = np.random.choice(N, N, replace=False)
				holos = [neuron_order[i*H: (i+1)*H] for i in range(int(np.ceil(N/H)))]
				for (rep, holo, power) in itertools.product(range(nreps), holos, powers):
					if K >= trials:
						break
					stim_trial = np.zeros(N)
					stim_trial[holo] = power
					stim_matrix += [stim_trial]
					K += 1

			reorder = np.random.choice(K, K, replace=False)
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
	
	print('Complete.')
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