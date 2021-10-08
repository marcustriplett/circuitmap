import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import time
import scipy.signal as sg

class NeuralDenoiser():
	def __init__(self, path=None, n_layers=3, kernel_size=99, padding=49, stride=1, channels=[16, 8, 1]):
		if path is not None:
			self.denoiser = torch.load(path)
		else:
			self.denoiser = DenoisingNetwork(n_layers=n_layers, kernel_size=kernel_size,
				padding=padding, stride=stride, channels=channels)

		self.denoiser.to("cuda" if torch.cuda.is_available() else "cpu")

	def __call__(self, traces, monotone_filter_start=500, monotone_filter_inplace=True, rescale=20):
		''' Run denoiser over PSC trace batch and apply monotone decay filter.
		'''
		den = self.denoiser(rescale * torch.Tensor(traces.copy()[:, None, :])).detach().numpy().squeeze()/rescale
		return _monotone_decay_filter(den, inplace=monotone_filter_inplace, 
			monotone_start=monotone_filter_start)

	def train(self, epochs=1000, batch_size=64, learning_rate=1e-2, data_path=None, save_every=50, 
		save_path=None):
		'''Run pytorch training loop.
		'''

		print('CUDA device available: ', torch.cuda.is_available())
		print('CUDA device: ', torch.cuda.get_device_name())

		if data_path is not None:
			print('Attempting to load data at', data_path, '... ', end='')
			_data = np.load(data_path)
			training_inputs = _data['training_inputs']
			training_targets = _data['training_targets']
			test_inputs = _data['test_inputs']
			test_targets = _data['test_targets']
			print('found.')
			
			# Torch format
			training_data = PSCData(training_inputs, training_targets)
			test_data = PSCData(test_inputs, test_targets)
		else:
			print('Attempting to load data from NeuralDenoiser object... ', end='')
			training_data = PSCData(self.training_data[0], self.training_data[1])
			test_data = PSCData(self.test_data[0], self.test_data[1])
			print('found.')

		train_dataloader = DataLoader(training_data, batch_size=batch_size)
		test_dataloader = DataLoader(test_data, batch_size=batch_size)

		loss_fn = nn.MSELoss()
		self.train_loss, self.test_loss = [], []
		optimizer = torch.optim.SGD(self.denoiser.parameters(), lr=learning_rate)

		# Run torch update loops
		print('Initiating neural net training...')
		t_start = time.time()
		for t in range(epochs):
			self.train_loss.append(_train_loop(train_dataloader, self.denoiser, loss_fn, optimizer))
			self.test_loss.append(_test_loop(test_dataloader, self.denoiser, loss_fn))
			print('Epoch %i/%i  Train loss: %.8f  Test loss: %.8f'%(t+1, epochs, self.train_loss[t], self.test_loss[t]))

			if (save_every is not None) and (t % save_every == 0) and (save_path is not None):
				torch.save(self.denoiser, save_path + '_chkpt_%i.pt'%t)
		t_stop = time.time()
		print("Training complete. Elapsed time: %.2f min."%((t_stop-t_start)/60))

	def generate_training_data(self, trial_dur=900, size=1000, training_fraction=0.9, lp_cutoff=500, 
		srate=2000, tau_r_lower=2, tau_r_upper=80, tau_diff_lower=2, tau_diff_upper=150, 
		delta_lower=160, delta_upper=400, next_delta_lower=400, next_delta_upper=899,
		mode_probs=[0.3, 0.5, 0.1, 0.1], noise_std_lower=0.01, noise_std_upper=0.1, 
		gp_lengthscale=25, gp_scale=0.01, max_modes=4, observed_amplitude_lower=0.75, 
		observed_amplitude_upper=1.25, save_path=None):
		'''Simulate data for training a PSC denoiser. 
		'''

		n_modes = np.random.choice(max_modes, size, p=mode_probs)
		n_modes_prev = np.random.choice(max_modes, size, p=mode_probs)
		n_modes_next = np.random.choice(max_modes, size, p=mode_probs)
		targets = np.zeros((size, trial_dur))
		prev_pscs = np.zeros((size, trial_dur))
		next_pscs = np.zeros((size, trial_dur))
		noise_stds = np.random.uniform(noise_std_lower, noise_std_upper, size)
		iid_noise = np.zeros((size, trial_dur))
		gp_noise = _sample_gp(n_samples=size, trial_dur=trial_dur, gp_lengthscale=gp_lengthscale,
			gp_scale=gp_scale)

		# generate PSC traces
		for i in range(size):
			# target PSCs initiate between 100 and 300 frames (5-15ms after trial onset)
			targets[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
				tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
				delta_lower=delta_lower, delta_upper=delta_upper, n_samples=n_modes[i]), 0)

			next_pscs[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
				tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
				delta_lower=next_delta_lower, delta_upper=next_delta_upper, 
				n_samples=n_modes_next[i]), 0)

			prev_pscs[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
				tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
				delta_lower=-400, delta_upper=-100, n_samples=n_modes_prev[i]), 0)
			
			iid_noise[i] = np.random.normal(0, noise_stds[i], trial_dur)

		# lowpass filter inputs as with experimental data
		b_lp, a_lp = sg.butter(4, lp_cutoff, btype='low', fs=srate)
		inputs = sg.filtfilt(b_lp, a_lp, prev_pscs + targets + next_pscs + gp_noise + iid_noise, axis=-1)
		ampl = np.random.uniform(observed_amplitude_lower, observed_amplitude_upper, size)[:, None]
		maxv = np.max(inputs, 1)[:, None]
		inputs = inputs/maxv * ampl
		targets = targets/maxv * ampl

		# save training data in object
		training_trials = int(training_fraction * size)
		self.training_data = (inputs[:training_trials], targets[:training_trials])
		self.test_data = (inputs[training_trials:], targets[training_trials:])

		if save_path is not None:
			if save_path[-1] != '/': save_path += '/'
			np.savez(save_path + 'neural_denoising_sim_data_td_%i_sz_%i.npy'%(trial_dur, size), 
				training_inputs=inputs[:training_trials], training_targets=targets[:training_trials], 
				test_inputs=inputs[training_trials:], test_targets=inputs[training_trials:])

class PSCData(Dataset):
	"""Torch training dataset.
	"""
	def __init__(self, inputs, targets):
		n_samples, trial_len = inputs.shape
		inputs = inputs.copy().reshape([n_samples, 1, trial_len])
		targets = targets.copy().reshape([n_samples, 1, trial_len])
		self.x_data = [torch.Tensor(row) for row in inputs]
		self.y_data = [torch.Tensor(row) for row in targets]
		self.len = inputs.shape[0]
		
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
		
	def __len__(self):
		return self.len

class DenoisingNetwork(nn.Module):
	def __init__(self, n_layers=3, kernel_size=99, padding=49, channels=[16, 8, 1], stride=1):
		super(DenoisingNetwork, self).__init__()
		assert n_layers >= 2, 'Neural network must have at least one input layer and one output layer.'
		assert channels[-1] == 1, 'Output layer must have exactly one output channel'

		layers = [nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=kernel_size, 
			stride=stride, padding=padding)]
		layers.append(nn.ReLU())

		for l in range(1, n_layers):
			layers.append(nn.Conv1d(in_channels=channels[l - 1], out_channels=channels[l], 
			kernel_size=kernel_size, stride=stride, padding=padding))
			layers.append(nn.ReLU())

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

def _sample_gp(trial_dur=800, gp_lengthscale=25, gp_scale=0.01, n_samples=1):
	D = np.array([[i - j for i in range(trial_dur)] for j in range(trial_dur)])
	K = np.exp(-D**2/(2 * gp_lengthscale**2))
	mean = np.zeros(trial_dur)
	return gp_scale * np.random.multivariate_normal(mean, K, size=n_samples)

def _kernel_func(tau_r, tau_d, delta):
	return lambda x: (np.exp(-(x - delta)/tau_d) - np.exp(-(x - delta)/tau_r)) * (x >= delta)

def _sample_psc_kernel(trial_dur=900, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=50, 
	tau_diff_upper=150, delta_lower=100, delta_upper=200, n_samples=1,
	amplitude_lower=0.1, amplitude_upper=1.5):
	'''Sample PSCs with random time constants, onset times, and amplitudes.
	'''
	if n_samples == 0:
		return np.zeros((1, trial_dur))
	tau_r_samples = np.random.uniform(tau_r_lower, tau_r_upper, n_samples)
	tau_diff_samples = np.random.uniform(tau_diff_lower, tau_diff_upper, n_samples)
	tau_d_samples = tau_r_samples + tau_diff_samples
	delta_samples = np.random.uniform(delta_lower, delta_upper, n_samples)
	xeval = np.arange(trial_dur)
	pscs = np.array([_kernel_func(tau_r_samples[i], tau_d_samples[i], delta_samples[i])(xeval) 
		for i in range(n_samples)])
	max_vec = np.max(pscs, 1)[:, None]
	amplitude = np.random.uniform(amplitude_lower, amplitude_upper, n_samples)[:, None]
	return pscs/max_vec * amplitude

def _monotone_decay_filter(arr, monotone_start=500, inplace=True):
	'''Enforce monotone decay beyond kwarg monotone_start. Performed in-place by default.
	'''
	if inplace:
		for t in range(monotone_start, arr.shape[1]):
			arr[:, t] = np.min([arr[:, t], arr[:, t-1]], axis=0)
		return arr
	else:
		_arr = np.copy(arr)
		for t in range(monotone_start, arr.shape[1]):
			_arr[:, t] = np.min([arr[:, t], _arr[:, t-1]], axis=0)
	return _arr

def _train_loop(dataloader, model, loss_fn, optimizer):
	n_batches = len(dataloader)
	train_loss = 0
	for batch, (X, y) in enumerate(dataloader):
		# sending the batch to GPU
		X.to("cuda" if torch.cuda.is_available() else "cpu")
		y.to("cuda" if torch.cuda.is_available() else "cpu")
		# Compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y)
		train_loss += loss

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss /= n_batches
	return train_loss.detach().numpy()
		
def _test_loop(dataloader, model, loss_fn):
	n_batches = len(dataloader)
	test_loss = 0
	with torch.no_grad():
		for X, y in dataloader:
			X.to("cuda" if torch.cuda.is_available() else "cpu")
			y.to("cuda" if torch.cuda.is_available() else "cpu")
			
			pred = model(X)
			test_loss += loss_fn(pred, y).item()

	test_loss /= n_batches
	return test_loss


