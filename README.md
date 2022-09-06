# CAVIaR

Computational tools for inferring synaptic connectivity from two-photon holographic neural ensemble stimulation. Accompanies the paper:
> _Rapid learning of neural circuitry from holographic ensemble stimulation enabled by Bayesian model-based compressed sensing_. (2022). M. A. Triplett\*, M. Gajowa\*, B. Antin, M. Sadahiro, H. Adesnik, and L. Paninski.

<img width="150" img align="right" alt="logo" src="https://user-images.githubusercontent.com/23161252/188723534-4674e1d9-202a-46ce-aeb6-22bb373a34e2.png"> 

Software developed by Marcus Triplett in the Paninski Lab at Columbia University, with contributions from Benjamin Antin.   

# Installation

First, clone and install the `circuitmap` package:

```
conda create -n circuitmap  
conda activate circuitmap  
cd path/to/install/location/  
git clone https://github.com/marcustriplett/circuitmap  
pip install ./circuitmap  
```

Then install the JAX version compatible with your OS and hardware using the instructions at https://github.com/google/jax.  
  
# Basic usage

First load (1) your `Kx900` matrix of PSC traces `psc`, and (2) the corresponding `NxK` stimulus design matrix `stim_matrix`. Each element of `stim_matrix` should provide the power delivered to neuron `n` on trial `k`.  

Load a pretrained NWD network (or train your own, see below):
```python
from circuitmap import NeuralDemixer
device = 'cpu' # optionally put device='cuda' for fast GPU demixing, but be aware that this can raise memory conflicts between PyTorch and JAX.
demix = NeuralDemixer(path='demixers/nwd_ie_ChroME2f.ckpt', device=device)
```

Demix the PSC traces with a single forward pass through the network:
```python
psc_dem = demix(psc)
```

Next, initialise and fit a statistical model using CAVIaR:
```python
import circuitmap as cm
model = cm.Model(N) 
model.fit(psc_dem, stim_matrix, method='caviar')
```

Once CAVIaR has completed, the inferred parameters will be stored in the `model.state` class attribute. The synaptic weights and presynaptic spikes can be extracted from the state via
```python
weights = model.state['mu'] # synaptic weight posterior mean
spikes = model.state['lam'] # spikes posterior mean
```

## Usage tips

`circuitmap` models come initialised with a set of priors that have been suitable for our simulations and experimental data. If custom priors are desired, they can be explicitly set via the `priors` kwarg during model initialisation:
```python
priors = {
  'alpha': 1/4 * np.ones(N),  # prior connection probability, used for CAVI-SnS only
  'phi': np.c_[1e-1 * np.ones(N), 5e0 * np.ones(N)],  # modes of power curve sigmoid coefficients
  'phi_cov': np.array([np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])),   # power curve prior covariances
  'mu': np.zeros(N),  # synaptic weight prior mean
  'beta': 1e1 * np.ones(N),   # synaptic weight prior standard deviation
  'shape': 1.,  # shape and rate parameters for gamma-distributed noise
  'rate': 1e-1
}

model = cm.Model(N, priors=priors)
```

Optimisation using CAVIaR can be fine-tuned using an optional `fit_options` dictionary supplied to the `model.fit()` routine. For example, some of the more important settings that might be adjusted to improve model fit are:
```python
fit_options = {
  'msrmp': 0.4, # default 0.3
  'iters': 30, # default 50
  'minimum_spike_count': 4, # default 3,
  'save_histories': True # default False
}

model.fit(psc_dem, stim_matrix, method='caviar', fit_options=fit_options)
```
The most critical parameter for adjusting model fit is the "min-spike-rate-at-max-power" variable `msrmp`. This determines how often the putative presynaptic cell must spike when stimulated at the maximum power used in the experiment for the synapse to be considered legitimate. For example, with `msrmp=0.4`, a presynaptic cell has to spike at least 40% of the time at max power to be considered connected. However, to actually evaluate whether this criterion is met, CAVIaR automatically performs an isotonic regression through the inferred presynaptic spikes using the classical pool-adjacent-violators-algorithm (PAVA) and inspects the value of the regressor at max power. Usage of PAVA reduces sensitivity to spontaneous spikes.

The variables `iters`, `minimum_spike_count`, and `save_histories` are respectively the number of iterations of the CAVIaR algorithm, the minimum number of spikes that a cell must emit overall, and whether or not the intermediate parameter estimates are saved. Leaving `'save_histories': False` considerably speeds up inference since otherwise intermediate parameters must be saved to device arrays on CPU. Putting `'save_histories': True` can be useful for trouble-shooting convergence issues, however.

## Available neural waveform demixer networks
Our suggested demixers (found in the `./demixers/` subdirectory) are `nwd_ie_ChroME2f.ckpt` for I-E mapping experiments using ChroME2f and `nwd_ee_ChroME1.ckpt` for E-E mapping experiments using ChroME. In rare cases, we found that it was useful to have a low-latency I-E variant `nwd_ie_ChroME2f_low_latency.ckpt` for when postsynaptic events arrive in less than 3 milliseconds.

## Training a new demixer
If the pretrained demixers are inadequate for your application (e.g., if the demixed PSCs decay too early or late) you may need to tailor the simulated training data to your application.

First, initialise a fresh NWD network without supplying a path:
```python
demix = NeuralDemixer()
```
Then adjust the relevant parameters of the training data and call the `train` routine. E.g.:
```python
tau_r_lower = 10 # upper and lower bounds on PSC rise times
tau_r_upper = 40

tau_diff_lower = 60 # upper and lower bounds on the difference between rise and decay times (this roundabout step ensures that the decay constant is larger than the rise constant)
tau_diff_upper = 120

demixer.generate_training_data(tau_r_lower=tau_r_lower, tau_r_upper=tau_r_upper, 
  tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper)
  
demixer.train(epochs=3000)
```
An example script for training a new NWD network can be found at `./scripts/train_psc_demixer.py`.  

## Other methods
We also include implementations of CAVI-SnS (based on Shababo et al. (2013)) and CoSaMP (from the `mr_utils` package by N. McKibben (2019); see https://github.com/mckib2/mr_utils/blob/master/mr_utils/cs/greedy/cosamp.py for the original implementation). These can be specified by setting `method='cavi_sns'` or `method='cosamp'` when calling `model.fit()`.

# Colab notebook demo  
Click [here](https://bit.ly/3wQcnVu) for a Google Colab notebook showing usage of NWD and CAVIaR on simulated connectivity mapping experiments.  

# Contact
Questions and feedback about the code can be directed to marcus.triplett@columbia.edu.
