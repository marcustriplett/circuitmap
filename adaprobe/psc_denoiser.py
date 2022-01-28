import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import numpy as np
import time
import scipy.signal as sg

# Conditionally import progress bar
try:
	get_ipython()
	from tqdm.notebook import tqdm
except:
	from tqdm import tqdm

class NeuralDenoiser():
	def __init__(self, path=None, eval_mode=True):
		# Set device dynamically
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Load or initialise denoiser object
		if path is not None:
			self.denoiser = DenoisingUNet().load_from_checkpoint(path)
			if eval_mode:
				self.denoiser.eval()
		else:
			self.denoiser = DenoisingUNet()

		# Move denoiser to device
		self.denoiser = self.denoiser.to(self.device)

	def __call__(self, traces, monotone_filter_start=500, monotone_filter_inplace=True, verbose=True):
		''' Run denoiser over PSC trace batch and apply monotone decay filter.
		'''

		if verbose: print('Demixing PSC traces... ', end='')
		t1 = time.time()

		tmax = np.max(traces, axis=1)[:, None]
		den = self.denoiser(
			torch.Tensor((traces/tmax).copy()[:, None, :]).to(device=self.device)
		).cpu().detach().numpy().squeeze() * tmax

		den = _monotone_decay_filter(den, inplace=monotone_filter_inplace, 
			monotone_start=monotone_filter_start)

		t2 = time.time()
		if verbose: print('complete (elapsed time %.2fs).'%(t2 - t1))

		return den

	def train(self, epochs=1000, batch_size=64, learning_rate=1e-2, data_path=None, save_every=50, 
		save_path=None, num_workers=2, pin_memory=True, num_gpus=1):
		'''Run pytorch training loop.
		'''

		# print('CUDA device available: ', torch.cuda.is_available())
		# print('CUDA device: ', torch.cuda.get_device_name())

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

		train_dataloader = DataLoader(training_data, batch_size=batch_size, 
			pin_memory=pin_memory, num_workers=num_workers)
		test_dataloader = DataLoader(test_data, batch_size=batch_size, 
			pin_memory=pin_memory, num_workers=num_workers)

		# Run torch update loops
		print('Initiating neural net training...')
		t_start = time.time()
		self.trainer = pl.Trainer(gpus=num_gpus, max_epochs=epochs)
		self.trainer.fit(self.denoiser, train_dataloader, test_dataloader)
		t_stop = time.time()

		print("Training complete. Elapsed time: %.2f min."%((t_stop-t_start)/60))

	def generate_training_data(self, trial_dur=900, size=1000, training_fraction=0.9, lp_cutoff=500, 
		srate=20000, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=2, tau_diff_upper=150, 
		delta_lower=160, delta_upper=400, next_delta_lower=400, next_delta_upper=899,
		prev_delta_lower=-400, prev_delta_upper=-100, mode_probs=[0.4, 0.4, 0.1, 0.1],
		prev_mode_probs=[0.5, 0.4, 0.05, 0.05], next_mode_probs=[0.5, 0.4, 0.05, 0.05],
		noise_std_lower=0.01, noise_std_upper=0.1, gp_lengthscale=25, gp_scale=0.01, 
		max_modes=4, observed_amplitude_lower=0.75, observed_amplitude_upper=1.25, 
		prob_zero_event=0.001, templates=None, template_prob=0.075, save_path=None):
		'''Simulate data for training a PSC denoiser. 
		'''

		n_modes = np.random.choice(max_modes, size, p=mode_probs)
		n_modes_prev = np.random.choice(max_modes, size, p=prev_mode_probs)
		n_modes_next = np.random.choice(max_modes, size, p=next_mode_probs)
		targets = np.zeros((size, trial_dur))
		prev_pscs = np.zeros((size, trial_dur))
		next_pscs = np.zeros((size, trial_dur))
		inputs = np.zeros((size, trial_dur))
		noise_stds = np.random.uniform(noise_std_lower, noise_std_upper, size)
		iid_noise = np.zeros((size, trial_dur))

		b_lp, a_lp = sg.butter(4, lp_cutoff, btype='low', fs=srate)

		# generate PSC traces
		for i in tqdm(range(size), desc='Trace generation', leave=True):
			# target PSCs initiate between 100 and 300 frames (5-15ms after trial onset)

			if (templates is not None) and (np.random.rand() <= template_prob):
				inputs[i] = templates[np.random.choice(templates.shape[0], 1)]
				targets[i] = np.zeros(templates.shape[1])

			else:
				targets[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
								tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
								delta_lower=delta_lower, delta_upper=delta_upper, n_samples=n_modes[i]), 0)

				next_pscs[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
								tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
								delta_lower=next_delta_lower, delta_upper=next_delta_upper, n_samples=n_modes_next[i]), 0)

				prev_pscs[i] = np.sum(_sample_psc_kernel(trial_dur=trial_dur, tau_r_lower=tau_r_lower, 
								tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper, 
								delta_lower=prev_delta_lower, delta_upper=prev_delta_upper, n_samples=n_modes_prev[i]), 0)

				# lowpass filter inputs as with experimental data
				inputs[i] = sg.filtfilt(b_lp, a_lp, prev_pscs[i] + targets[i] + next_pscs[i], axis=-1)
			
			iid_noise[i] = np.random.normal(0, noise_stds[i], trial_dur)

		gp_noise = _sample_gp(n_samples=size, trial_dur=trial_dur, gp_lengthscale=gp_lengthscale,
			gp_scale=gp_scale) * np.random.uniform(0, 1, size)[:, None]

		maxv = np.max(inputs, 1)[:, None] + 1e-5
		inputs = inputs/maxv + gp_noise + iid_noise
		targets = targets/maxv

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

class StackedDenoisingNetwork(nn.Module):
	'''Denoising neural network consisting of multiple layers of long 1d convolutions.
	'''
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

class DownsamplingBlock(nn.Module):
	'''DownsamplingBlock
	'''
	def __init__(self, in_channels, out_channels, kernel_size, dilation):
		super(DownsamplingBlock, self).__init__()

		self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
			dilation=dilation)
		self.decimate = nn.AvgPool1d(kernel_size=3, stride=2)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_channels)

	def forward(self, x):
		return self.relu(self.bn(self.conv(self.decimate(x))))

class UpsamplingBlock(nn.Module):
	'''UpsamplingBlock
	'''
	def __init__(self, in_channels, out_channels, kernel_size, stride, interpolation_mode='linear'):
		super(UpsamplingBlock, self).__init__()

		self.deconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, 
			kernel_size=kernel_size, stride=stride)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_channels)
		self.interpolation_mode = interpolation_mode

	def forward(self, x, skip=None, interp_size=None):
		if skip is not None:
			up = nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=skip.shape[-1], 
										  mode=self.interpolation_mode, align_corners=False)
			return torch.cat([up, skip], dim=1)
		else:
			return nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=interp_size, 
				mode=self.interpolation_mode, align_corners=False)

class ConvolutionBlock(nn.Module):
	'''ConvolutionBlock
	'''
	def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
		super(ConvolutionBlock, self).__init__()
		
		self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
							  stride=stride, padding=padding, dilation=dilation)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_channels)
		
	def forward(self, x):
		return self.relu(self.bn(self.conv(x)))

class DenoisingUNet(pl.LightningModule):
	def __init__(self):
		super(DenoisingUNet, self).__init__()
		self.dblock1 = DownsamplingBlock(1, 16, 32, 2)
		self.dblock2 = DownsamplingBlock(16, 16, 32, 1)
		self.dblock3 = DownsamplingBlock(16, 32, 16, 1)
		self.dblock4 = DownsamplingBlock(32, 32, 16, 1)
		
		self.ublock1 = UpsamplingBlock(32, 16, 16, 1)
		self.ublock2 = UpsamplingBlock(48, 16, 16, 1)
		self.ublock3 = UpsamplingBlock(32, 16, 32, 1)
		self.ublock4 = UpsamplingBlock(32, 4, 32, 2)
		
		self.conv = ConvolutionBlock(4, 1, 256, 255, 1, 2)
		
	def forward(self, x):
		# Encoding
		enc1 = self.dblock1(x)
		enc2 = self.dblock2(enc1)
		enc3 = self.dblock3(enc2)
		enc4 = self.dblock4(enc3)

		# Decoding
		dec1 = self.ublock1(enc4, skip=enc3)
		dec2 = self.ublock2(dec1, skip=enc2)
		dec3 = self.ublock3(dec2, skip=enc1)
		dec4 = self.ublock4(dec3, interp_size=x.shape[-1])

		# Final conv layer
		out = self.conv(dec4)

		return out

	def configure_optimizers(self):
		return torch.optim.SGD(self.parameters(), lr=1e-2)
	
	def loss_fn(self, inputs, targets):
		return nn.functional.mse_loss(inputs, targets)
	
	def training_step(self, batch, batch_idx):
		x, y = batch
		pred = self.forward(x)
		loss = self.loss_fn(pred, y)
		self.log('train_loss', loss)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y = batch
		pred = self.forward(x)
		loss = self.loss_fn(pred, y)
		self.log('val_loss', loss)

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
