import numpy as np
import matplotlib.pyplot as plt

DEFAULT_OMEGA = np.array([0.00389425, 0.00391111, 0.00074478])
DEFAULT_PHI = np.array([0.03203156, 5.216092])

class Simulation3d:
	def __init__(self, dimx=125, dimy=125, dimz=100, spacing=10, grid_density=5, N=16, a=0.5, sigma=3, phi_0=None,
		phi_1=None, min_w=3, max_w=20, mode='online', spont_prob=0.05, spont_mean=5, spont_max=20, multiplicative_std=0.1,
		min_mult_noise=0.05):
		""" Initialise a 3d adaprobe simulation object.
		"""
		self.gridx, self.gridy, self.gridz, self.xr, self.yr, self.zr = _generate_3d_grid(dimx, dimy, dimz, grid_density)
		self.grid = np.c_[self.gridx.flatten(), self.gridy.flatten(), self.gridz.flatten()]
		self.cell_locs = _generate_3d_locs(dimx, dimy, dimz, N, spacing)

		z, u = _generate_weights(N, a, min_w=min_w, max_w=max_w)
		self.z = z
		self.u = u
		self.w = u * z
		self.N = N
		self.sigma = sigma

		self.spont_prob = spont_prob
		self.spont_mean = spont_mean
		self.spont_max = spont_max

		self.multiplicative_std = multiplicative_std
		self.min_mult_noise = min_mult_noise

		if phi_0 is None:
			self.phi_0 = np.random.uniform(0.1, 0.2, N)
		else:
			self.phi_0 = phi_0

		if phi_1 is None:
			self.phi_1 = np.random.uniform(2.5, 5.5, N)
		else:
			self.phi_1 = phi_1

		self.reset()

	def reset(self):
		"""Reset trials.
		"""
		self.tars = []
		self.I = []
		self.fr = []
		self.spks = []
		self.mult_noise = []
		self.spont = []
		self.y = []
		self.trials = 0
		return

	def next_trial(self, tar, I):
		""" Simulate next trial at neuron n with power I.
		"""
		cell_inds = np.arange(self.N)
		fr = _sigmoid(self.phi_0 * I * (cell_inds == tar) - self.phi_1) * (cell_inds == tar)
		spks = np.random.rand(self.N) <= fr
		mult_noise = np.random.normal(1, self.multiplicative_std, self.N) # multiplicative noise
		mult_noise[mult_noise < self.min_mult_noise] = self.min_mult_noise # prevent multipliers <= 0
		spont = (np.random.rand() <= self.spont_prob) * np.random.exponential(self.spont_mean) # spontaneous effects
		spont = np.min([spont, self.spont_max])
		y = np.random.normal(self.w @ (mult_noise * spks), self.sigma) - spont

		# Save simulation result to object
		self.tars += [tar]
		self.I += [I]
		self.fr += [fr]
		self.spks += [spks]
		self.mult_noise += [mult_noise]
		self.spont += [spont]
		self.y += [y]
		self.trials += 1
		return

	def simulate(self, trials=1000, powers=None):
		"""Simulate fixed number of trials.
		"""

		# Reset sim
		self.reset()

		# Configure available laser powers
		if powers is None:
			powers = np.arange(10, 51, 10) # default power range

		# Run trials
		for k in range(trials):
			tar = np.mod(k, self.N)
			power = np.random.choice(powers)
			self.next_trial(tar, power)

		self.tars = np.array(self.tars)
		self.I = np.array(self.I)
		self.y = np.array(self.y)
		self.fr = np.array(self.fr).T
		self.spks = np.array(self.spks).T
		self.mult_noise = np.array(self.mult_noise).T
		self.spont = np.array(self.spont)
		return

	def next_trial_multistim(self, tars, I):
		""" Simulate next trial at set of neurons tars with power I.
		"""
		cell_inds = np.arange(self.N)
		I_multi = np.zeros(self.N)
		I_multi[tars] = I
		fr = _sigmoid(self.phi_0 * I_multi - self.phi_1) * (I_multi > 0)
		spks = np.random.rand(self.N) <= fr
		mult_noise = np.random.normal(1, self.multiplicative_std, self.N) # multiplicative noise
		mult_noise[mult_noise < self.min_mult_noise] = self.min_mult_noise # prevent multipliers <= 0
		spont = (np.random.rand() <= self.spont_prob) * np.random.exponential(self.spont_mean) # spontaneous effects
		spont = np.min([spont, self.spont_max])
		y = np.random.normal(self.w @ (mult_noise * spks), self.sigma) - spont

		# Save simulation result to object
		self.tars += [tars]
		self.I += [I]
		self.fr += [fr]
		self.spks += [spks]
		self.mult_noise += [mult_noise]
		self.spont += [spont]
		self.y += [y]
		self.trials += 1
		return

	def simulate_multistim(self, trials=1000, num_targets=4, powers=None):
		# Reset sim
		self.reset()

		# Configure available laser powers
		if powers is None:
			powers = np.arange(10, 51, 10) # default power range

		# Run trials
		for k in range(trials):
			tars = np.random.choice(self.N, num_targets, replace=False)
			power = np.random.choice(powers)
			self.next_trial_multistim(tars, power)

		self.tars = np.array(self.tars)
		self.I = np.array(self.I)
		self.y = np.array(self.y)
		self.fr = np.array(self.fr).T
		self.mult_noise = np.array(self.mult_noise).T
		self.spont = np.array(self.spont)
		self.spks = np.array(self.spks).T
		return

class Simulation2d:
	def __init__(self, layout='grid', dimension=2, dim2d=None, spacing=10, N=16, a=0.75, b=10, sigma=3, omega=None, phi_0=None, phi_1=None, rho=1e-2, min_w=3, max_w=20, 
		grid_density=1,	mode='online'):
		"""Initialise an adaprobe simulation object.
		"""
		assert layout in ['linear', 'grid', 'random'], """Kwarg 'layout' must be either 'linear', 'grid' or 'random'."""
		self.layout = layout
		self.spacing = spacing # in um
		if dim2d is None:
			# generate grid from cell_locs
			self.cell_locs = _generate_locs(N, layout, spacing)
			self.gridx, self.gridy, self.xr, self.yr = _generate_grid(self.cell_locs, spacing, grid_density) 
		else:
			# generate cell_locs from grid
			self.cell_locs, self.gridx, self.gridy, self.xr, self.yr = _generate_grid_with_given_dim(dim2d, N, spacing=spacing) 

		# Configure default params for neural population
		self.sigma = sigma
		self.N = N

		if omega is None:
			omega = 1e-2 * np.ones(N)
		self.omega = omega

		z, u = _generate_weights(N, a, max_w=max_w)
		self.u = u
		self.z = z
		self.w = u * z

		if phi_0 is None:
			phi_0 = np.random.uniform(0.08, 0.15, N)
		if phi_1 is None:
			half_prob = np.random.uniform(70, 90, N)
			phi_1 = phi_0 * half_prob
		self.phi_0 = phi_0
		self.phi_1 = phi_1
		self.rho = rho

		if mode == 'online':
			self.L = []
			self.I = []
			self.fr = []
			self.spks = []
			self.y = []
			self.spike_noise = []

		self.trials = None

	def reset(self):
		"""Reset simulation trials.
		"""
		self.L = []
		self.I = []
		self.fr = []
		self.spks = []
		self.spike_noise = []
		self.y = []
		self.trials = 0

	def next_trial(self, L, I):
		"""Simulate a single trial. Location can be either a user-provided 2d-coordinate, a random 
		location, or a random soma.
		"""

		fr = _sigmoid(self.phi_0 * I * np.exp(-self.omega * np.sum(np.square(L - self.cell_locs), 1)) - self.phi_1)
		spks = np.random.rand(self.N) <= fr
		y = np.random.normal(self.w @ spks, self.sigma)

		# Save simulation result to object
		self.L += [L]
		self.I += [I]
		self.fr += [fr]
		self.spks += [spks]
		self.y += [y]

	def next_trial_multistim(self, L, I):
		"""Simulate a single trial. Location can be either a user-provided 2d-coordinate, a random 
		location, or a random soma.
		"""

		mk = np.array([np.sum(I * np.exp(-self.omega[n] * np.sum(np.square(L - self.cell_locs[n]), 1))) for n in range(self.N)])
		fr = _sigmoid(self.phi_0 * mk - self.phi_1)
		spks = np.random.rand(self.N) <= fr
		y = np.random.normal(self.w @ spks, self.sigma)

		# Save simulation result to object
		self.L += [L]
		self.I += [I]
		self.fr += [fr]
		self.spks += [spks]
		self.y += [y]

	def next_trial_spike_noise(self, L, I):
		fr = _sigmoid(self.phi_0 * I * np.exp(-self.omega * np.sum(np.square(L - self.cell_locs), 1)) - self.phi_1)
		spks = np.random.rand(self.N) <= fr
		no_spike_locs = np.where(1 - spks)[0]
		spike_noise = np.random.normal(1, self.rho, self.N) * spks
		spike_noise[no_spike_locs] = 1
		y = np.random.normal(np.sum(self.w * spike_noise * spks), self.sigma)

		# Save simulation result to object
		self.L += [L]
		self.I += [I]
		self.fr += [fr]
		self.spike_noise += [spike_noise]
		self.spks += [spks]
		self.y += [y]

	def simulate(self, trials=100, burnin=0, powers=None, design='ordered', jitter=1):
		"""Simulate fixed number of trials with the given experimental design.
		"""

		if design == 'ordered':
			self._simulate_ordered(trials, powers, jitter=jitter) # no changes needed for burnin with an already ordered design.
		elif design == 'random':
			self._simulate_random(trials, burnin, powers)
		elif design == 'random_grid':
			self._simulate_random_grid(trials, burnin, powers)
		else:
			raise Exception("""Kwarg 'design' must be either 'ordered' for ordered
		 soma-targeted holograms, 'random' for uniformly random hologram targets, or 'random_grid'.""")

	def _simulate_ordered(self, trials, powers, jitter=1):
		"""Simulate mapping experiment with ordered soma-targeted holograms.
		"""

		# Set up vars
		spks = np.zeros((self.N, trials))
		y = np.zeros(trials)
		fr = np.zeros((self.N, trials))
		L = np.zeros((trials, 2))

		# Configure available laser powers
		if powers is None:
			powers = np.arange(80, 120, 10)

		I = np.random.choice(powers, trials, replace=True)

		# Run simulation
		for k in range(trials):
			L[k] = self.cell_locs[np.mod(k, self.N)] + np.random.normal(0, jitter, 2)
			fr[:, k] = _sigmoid(self.phi_0 * I[k] \
				* np.exp(-self.omega * np.sum(np.square(L[k] - self.cell_locs), 1)) - self.phi_1)
			spks[:, k] = np.random.rand(self.N) <= fr[:, k]
			y[k] = np.random.normal(self.w @ spks[:, k], self.sigma)

		# Save simulation results to object
		self.trials 	= trials
		self.L 			= list(L)
		self.I 			= list(I)
		self.fr 		= list(fr.T)
		self.spks 		= list(spks.T)
		self.y 			= list(y)

	def _simulate_random(self, trials, burnin, powers):
		"""Simulate mapping experiment with uniformly random hologram targets.
		"""

		# Set up vars
		spks = np.zeros((self.N, trials))
		y = np.zeros(trials)
		fr = np.zeros((self.N, trials))
		L = np.zeros((trials, 2))

		# Configure available laser powers
		if powers is None:
			powers = np.arange(80, 120, 10)

		I = np.random.choice(powers, trials, replace=True)

		# Run simulation
		for k in range(trials):
			if k < burnin:
				L[k] = self.cell_locs[np.mod(k, self.N)]
			else:
				L[k] = np.array([
					np.random.uniform(np.min(self.cell_locs[:, 0]) - self.spacing, 
						np.max(self.cell_locs[:, 0]) + self.spacing),
					np.random.uniform(np.min(self.cell_locs[:, 1]) - self.spacing,
						np.max(self.cell_locs[:, 1]) + self.spacing)
				])
			fr[:, k] = _sigmoid(self.phi_0 * I[k] \
				* np.exp(-self.omega * np.sum(np.square(L[k] - self.cell_locs), 1)) - self.phi_1)
			spks[:, k] = np.random.rand(self.N) <= fr[:, k]
			y[k] = np.random.normal(self.w @ spks[:, k], self.sigma)

		# Save simulation results to object
		self.trials 	= trials
		self.L 			= list(L)
		self.I 			= list(I)
		self.fr 		= list(fr.T)
		self.spks 		= list(spks.T)
		self.y 			= list(y)

	def _simulate_random_grid(self, trials, burnin, powers):
		"""Simulate mapping experiment with uniformly random hologram targets.
		"""

		# Set up vars
		spks = np.zeros((self.N, trials))
		y = np.zeros(trials)
		fr = np.zeros((self.N, trials))
		L = np.zeros((trials, 2))

		# Configure available laser powers
		if powers is None:
			powers = np.arange(80, 120, 10)

		I = np.random.choice(powers, trials, replace=True)

		grid_len = 26
		nsqrt = np.sqrt(self.N) * self.spacing

		# Run simulation
		for k in range(trials):
			if k < burnin:
				L[k] = self.cell_locs[np.mod(k, self.N)]
			else:
				L[k] = np.random.choice(np.arange(-self.spacing, nsqrt + self.spacing, (nsqrt + 2*self.spacing)/grid_len), 2, replace=True)
			fr[:, k] = _sigmoid(self.phi_0 * I[k] \
				* np.exp(-self.omega * np.sum(np.square(L[k] - self.cell_locs), 1)) - self.phi_1)
			spks[:, k] = np.random.rand(self.N) <= fr[:, k]
			y[k] = np.random.normal(self.w @ spks[:, k], self.sigma)

		# Save simulation results to object
		self.trials 	= trials
		self.L 			= list(L)
		self.I 			= list(I)
		self.fr 		= list(fr.T)
		self.spks 		= list(spks.T)
		self.y 			= list(y)

	def view_spike_prob_map(self, n, power=100, figsize=(4.5, 4), fontsize=12, save=None):
		grid = np.c_[self.gridx.flatten(), self.gridy.flatten()]
		spk_map = _sigmoid(self.phi_0[n] * power \
			* np.exp(-self.omega[n] * np.sum(np.square(self.cell_locs[n] - grid), 1)) - self.phi_1[n])
		plt.figure(figsize=figsize)
		plt.title('Spike probability map', fontsize=fontsize)
		plt.contourf(self.gridx, self.gridy, spk_map.reshape(len(self.yr), len(self.xr)), 50)
		plt.colorbar()
		plt.scatter(self.cell_locs[:, 0], self.cell_locs[:, 1], s=100, edgecolor='white', facecolor='None', linewidth=1)
		plt.scatter(self.cell_locs[:, 0], self.cell_locs[:, 1], marker='x', s=50, color='r', linewidth=1)
		if save is not None:
			plt.savefig(save, format='png', bbox_inches='tight')
		else:
			plt.show()

def _generate_3d_grid(dimx, dimy, dimz, grid_density):
	xr = np.arange(0, dimx + 1, grid_density)
	yr = np.arange(0, dimy + 1, grid_density)
	zr = np.arange(0, dimz + 1, grid_density)
	gridx, gridy, gridz = np.meshgrid(xr, yr, zr)
	return gridx, gridy, gridz, xr, yr, zr

def _generate_3d_locs(dimx, dimy, dimz, N, spacing, max_attempts=100):
	"""
	"""
	dims = [dimx, dimy, dimz]
	total_dims = 3
	C = np.zeros((N, total_dims))
	for i in range(total_dims):
		C[0, i] = np.random.uniform(0, dims[i])
	for n in range(1, N):
		while i < max_attempts:
			loc = np.array([np.random.uniform(0, dims[i]) for i in range(total_dims)])
			if len(np.where(np.sqrt(np.sum(np.square(C[:n] - loc), 1)) <= spacing)[0]) == 0:
				break # break while loop, continue with placing next cell
			i += 1
		C[n] = loc
	return C

def _generate_weights(N, a, min_w=3, max_w=20):
	z = np.random.rand(N) < a
	u = np.random.uniform(-max_w, -min_w, N)
	return z, u

def _generate_locs(N, layout, spacing):
	if layout == 'linear':
		C = np.array([[spacing * n, 0] for n in range(N)])
		return C
	elif layout == 'grid':
		nsqrt = int(np.sqrt(N))
		assert nsqrt**2 == N, """N must be square for layout of type 'grid'.""" 
		C = np.array([
			np.array([n * spacing, m * spacing]) for n in range(nsqrt) for m in range(nsqrt)
		])
		return C
	elif layout == 'random':
		# rejection sampling placement, occupies the sqrt(N)*spacing x sqrt(N)*spacing square
		nsqrt = int(np.ceil(np.sqrt(N)) * spacing)
		C = np.zeros((N, 2))
		max_attempts = 100 # number of placement attempts before giving up
		min_dist = 10 # microns
		C[0] = np.random.uniform(0, nsqrt, 2)
		for n in range(1, N):
			i = 0
			while i < max_attempts:
				loc = np.random.uniform(0, nsqrt, 2)
				if len(np.where(np.sqrt(np.sum(np.square(C[:n] - loc), 1)) < min_dist)[0]) == 0:
					break # break while loop, continue with placing next cell
				i += 1
			C[n] = loc
		return C

def _generate_grid(cell_locs, spacing, grid_density=1):
	"""Generate mesh grid encompassing cell locations.
	"""
	minx, maxx = np.min(cell_locs[:, 0]), np.max(cell_locs[:, 0])
	miny, maxy = np.min(cell_locs[:, 1]), np.max(cell_locs[:, 1])
	xr = np.arange(minx - spacing, maxx + spacing, grid_density)
	yr = np.arange(miny - spacing, maxy + spacing, grid_density)
	xgrid, ygrid = np.meshgrid(xr, yr)
	return xgrid, ygrid, xr, yr

def _generate_grid_with_given_dim(dim, N, spacing=10, border=5, max_attempts=100):
	"""Generate mesh grid with given dimension and fill with random cell locations.
	"""
	xr, yr = np.arange(0, dim), np.arange(0, dim)
	xgrid, ygrid = np.meshgrid(xr, yr)

	C = np.zeros((N, 2))
	C[0] = np.random.uniform(border, dim - border, 2)
	for n in range(1, N):
		i = 0
		while i < max_attempts:
			loc = np.random.uniform(border, dim - border, 2)
			if len(np.where(np.sqrt(np.sum(np.square(C[:n] - loc), 1)) < spacing)[0]) == 0:
				break # break while loop, continue with placing next cell
			i += 1
		C[n] = loc

	return C, xgrid, ygrid, xr, yr

def _sigmoid(x):
	return 1./(1. + np.exp(-x))
