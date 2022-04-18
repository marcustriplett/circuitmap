import argparse
import numpy as np
import circuitmap as cm
from circuitmap.simulation import simulate_continuous_experiment
from circuitmap import NeuralDemixer
from datetime import date
import _pickle as cpickle # pickle compression
from tqdm import tqdm
import bz2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--connection_prob')
	parser.add_argument('--spont_rate')
	parser.add_argument('--ntars')
	parser.add_argument('--stim_freq')
	parser.add_argument('--demixer')
	parser.add_argument('--token')
	parser.add_argument('--out')
	parser.add_argument('--n_sims')
	args = parser.parse_args()

	ntars 			= int(args.ntars)
	spont_rate 		= float(args.spont_rate)
	connection_prob = float(args.connection_prob)
	stim_freq 		= float(args.stim_freq)
	token 			= args.token
	out 			= args.out
	n_sims 			= int(args.n_sims)

	N = 300
	nreps = 1
	trials = 2000
	sampling_freq = 20000
	demix = NeuralDemixer(path=args.demixer)
	expt_len = np.ceil(trials/stim_freq) * sampling_freq
	ground_truth_eval_batch_size = 100

	results = {}

	for i in tqdm(range(n_sims), leave=True):
		sim = simulate_continuous_experiment(N=N, H=ntars, nreps=nreps, spont_rate=spont_rate, 
			connection_prob=connection_prob, stim_freq=stim_freq, expt_len=expt_len,
			ground_truth_eval_batch_size=ground_truth_eval_batch_size)

		results['trial_%i'%i] = {
			'true_responses': expt['true_responses'],
			'obs_responses': expt['obs_responses'],
			'demixed_responses': demix(expt['obs_responses'])
		}

	if out[-1] != '/': out += '/'

	np.savez_compressed(out + 'sim_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i'%(
		N, trials, ntars, nreps, connection_prob, spont_rate
	) + token + '.npz', results=results)