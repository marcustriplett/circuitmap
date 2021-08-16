import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

print('CUDA device available: ', torch.cuda.is_available())
print('CUDA device: ', torch.cuda.get_device_name())

class NeuralDenoiser():
	def __init__(self):
		self.denoiser = DenoisingNetwork()

	def __call__(self):

		return

	def train(epochs=1000, batch_size=64, learning_rate=1e-2, data_path=None):
		'''Run pytorch training loop.
		'''
		if data_path is not None:
			print('Attempting to load data at', data_path)
			_data = np.load(data_path)
			training_inputs = _data['training_inputs']
			training_targets = _data['training_targets']
			test_inputs = _data['test_inputs']
			test_targets = _data['test_targets']
			
			# Torch format
			training_data = PSCData(training_inputs, training_targets)
			test_data = PSCData(test_inputs, test_targets)
		else:
			print('Attempting to load data from NeuralDenoiser object')
			training_data = PSCData(self.training_data[0], self.training_Data[1])
			test_data = PSCData(self.test_data[0], self.test_data[1])

		train_dataloader = DataLoader(training_data, batch_size=batch_size)
		test_dataloader = DataLoader(test_data, batch_size=batch_size)

		loss_fn = nn.MSELoss()
		train_loss, test_loss = np.zeros(epochs), np.zeros(epochs)
		optimizer = torch.optim.SGD(self.denoiser.parameters(), lr=learning_rate)

		# Run torch update loops
		print('Initiating neural net training...')
		for t in range(epochs):
		    train_loss[t] = _train_loop(train_dataloader, self.denoiser, loss_fn, optimizer)
		    test_loss[t] = _test_loop(test_dataloader, self.denoiser, loss_fn)
		    print('Epoch %i/%i Train loss: %.8f.  Test loss: %.8f'%(t+1, epochs, train_loss[t], test_loss[t]))
		print("Training complete.")

		# Save training loss
		self.train_loss = train_loss
		self.test_loss = test_loss

	def _train_loop(dataloader, model, loss_fn, optimizer):
		n_batches = len(dataloader)
		train_loss = 0
		for batch, (X, y) in enumerate(dataloader):
			# Compute prediction and loss
			pred = model(X)
			loss = loss_fn(pred, y)
			train_loss += loss

			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		train_loss /= n_batches
		return train_loss
			
	def _test_loop(dataloader, model, loss_fn):
		n_batches = len(dataloader)
		test_loss = 0

		with torch.no_grad():
			for X, y in dataloader:
				pred = model(X)
				test_loss += loss_fn(pred, y).item()

		test_loss /= n_batches
		return test_loss

	def generate_training_data(trial_dur=800, size=1000, training_fraction=0.9, lp_cutoff=500, 
		srate=2000, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=50, tau_diff_upper=150, 
		min_delta=100, delta_lower=0, delta_upper=400, n_kernel_samples=1, 
		mode_probs=[0.3, 0.5, 0.1, 0.1], noise_std_lower=0.01, noise_std_upper=0.1, 
		gp_lengthscale=25, gp_scale=0.01, max_nodes=4, save_path=None):
		'''Simulate data for training a PSC denoiser. 
		'''

		n_modes, n_modes_prev = np.random.choice(max_nodes, size, p=mode_probs), \
			np.random.choice(max_nodes, size, p=mode_probs)
		targets, prev_pscs = np.zeros((size, trial_dur)), np.zeros((size, trial_dur))
		noise_stds = np.random.uniform(noise_std_lower, noise_std_upper)
		iid_noise = np.zeros((size, trial_dur))
		gp_noise = _sample_gp(n_samples=size, trial_dur=trial_dur, gp_lengthscale=gp_lengthscale,
			gp_scale=gp_scale)

		# generate PSC traces
		for i in range(size):
			targets[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
				tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
				min_delta=min_delta, delta_lower=delta_lower, delta_upper=delta_upper, 
				n_samples=n_modes[i]), 0)
			prev_pscs[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
				tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
				min_delta=min_delta, delta_lower=-400, delta_upper=-100, n_samples=n_modes[i]), 0)
			iid_noise[i] = np.random.normal(0, noise_stds[i], trial_dur)

		# lowpass filter inputs as with experimental data
		b_lp, a_lp = sg.butter(4, lp_cutoff, btype='low', fs=srate)
		inputs = sg.filtfilt(b_lp, a_lp, prev_pscs + targets + gp_noise + iid_noise, axis=-1)

		# save training data in object
		training_trials = int(training_fraction * size)
		self.training_data = (inputs[:training_trials], targets[:training_trials])
		self.test_data = (inputs[training_trials:], targets[training_trials:])

		if save_path is not None:
			if save_path[-1] != '/': save_path += '/'
			np.savez(save_path + 'neural_denoising_sim_data_td_%i_sz_%i.npy'%(trial_dur, size), 
				training_inputs=inputs[:training_trials], training_targets=targets[:training_trials], 
				test_inputs=inputs[training_trials:], test_targets=inputs[training_trials:])

	def _sample_gp(trial_dur=800, gp_lengthscale=25, gp_scale=0.01, n_samples=1):
		D = np.array([[i - j for i in range(trial_dur)] for j in range(trial_dur)])
		K = np.exp(-D**2/(2 * gp_lengthscale**2))
		mean = np.zeros(trial_dur)
		return gp_scale * np.random.multivariate_normal(mean, K, size=n_samples)

	def _kernel_func(tau_r, tau_d, delta):
		return lambda x: (np.exp(-(x - delta)/tau_d) - np.exp(-(x - delta)/tau_r)) * (x >= delta)

	def _sample_psc_kernel(trial_dur=800, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=50, 
		tau_diff_upper=150, min_delta=100, delta_lower=0, delta_upper=200, n_samples=1):
		'''Sample PSCs with random time constants and onset times.
		'''
		tau_r_samples = np.random.uniform(tau_r_lower, tau_r_upper, n_samples)
		tau_diff_samples = np.random.uniform(tau_diff_lower, tau_diff_upper, n_samples)
		tau_d_samples = tau_r_samples + tau_diff_samples
		delta_samples = min_delta + np.random.uniform(delta_lower, delta_upper, n_samples)
		xeval = np.arange(trial_dur)
		return np.array([_kernel_func(tau_r_samples[i], tau_d_samples[i], delta_samples[i])(xeval) 
			for i in range(n_samples)])

class PSCData(Dataset):
	"""Torch training dataset.
	"""
	def __init__(self, inputs, targets):
		n_samples, trial_len = inputs.shape
		inputs = inputs.copy().reshape([n_samples, 1, trial_len])
		targets = targets.copy().reshape([n_samples, 1, trial_len])
		self.x_data = [torch.from_numpy(row) for row in inputs]
		self.y_data = [torch.from_numpy(row) for row in targets]
		self.len = inputs.shape[0]
		
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]
		
	def __len__(self):
		return self.len

class DenoisingNetwork(torch.nn.Module):
	def __init__(self, kernel_size=99, padding=):
		super(PSCDenoiser, self).__init__()
		self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, 
			stride=stride, padding=padding, dilation=1)
		self.layer2 = torch.nn.Conv1d(in_channels=16, out_channels=8, kernel_size=kernel_size, 
			stride=stride, padding=padding, dilation=1)
		self.layer3 = torch.nn.Conv1d(in_channels=8, out_channels=1, kernel_size=kernel_size, 
			stride=stride, padding=padding, dilation=1)
		self.layers = [self.layer1, self.layer2, self.layer3]
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		for layer in self.layers:
			x = self.relu(layer(x))
		return x


