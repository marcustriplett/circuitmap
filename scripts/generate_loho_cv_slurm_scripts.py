import os
import h5py
import argparse
import subprocess
import numpy as np
from scipy.io import loadmat

def _generate_script_caviar(data_path, fname, njobs, demixer, msrmp, out, reader, start=0):
	script = "#!/bin/bash\n" +\
	"#SBATCH --job-name=loho_cv\n" +\
	"#SBATCH -c 1\n" +\
	"#SBATCH -o logs/slurm-%A_%a.out\n" +\
	"#SBATCH --mem-per-cpu=8gb\n" +\
	"#SBATCH --gres=gpu:1\n" +\
	"#SBATCH --exclude=ax[11-17]\n" +\
	"#SBATCH --array=0-" + str(njobs-1) + "\n" +\
	"ml anaconda3-2019.03 cuda/11.2.0 cudnn\n" +\
	"conda activate cuda11\n" +\
	"python circuitmap/scripts/run_loho_cv_caviar.py --data %s --demixer %s --out %s --msrmp %s --test_hologram_id $SLURM_ARRAY_TASK_ID --start %i --reader %s"%(
		data_path + fname, demixer, out, msrmp, start, reader
	)
	script_filename = "scripts/loho_cv_jobscripts/caviar/loho_cv_jobscript_caviar_%s_njobs%i_start%i.sh"%(fname.split('.')[0], njobs, start)
	save_and_call_script(script, script_filename)

def generate_script_caviar(data_path, fname, njobs, demixer, msrmp, out, reader, maxjobs=1000):
	start = 0
	remaining_jobs = njobs
	while remaining_jobs > 0:
		if remaining_jobs < maxjobs:
			# finished
			_generate_script_caviar(data_path, fname, remaining_jobs, demixer, msrmp, out, reader, start=start)
		else:
			# generate max feasible jobs and iterate
			_generate_script_caviar(data_path, fname, maxjobs, demixer, msrmp, out, reader, start=start)
		start += maxjobs
		remaining_jobs -= maxjobs

def generate_script_cavi_sns(data_path, fname, njobs, out, reader):
	script = "#!/bin/bash\n" +\
	"#SBATCH --job-name=loho_cv\n" +\
	"#SBATCH -c 1\n" +\
	"#SBATCH -o logs/slurm-%A_%a.out\n" +\
	"#SBATCH --mem-per-cpu=8gb\n" +\
	"#SBATCH --gres=gpu:1\n" +\
	"#SBATCH --exclude=ax[11-17]\n" +\
	"#SBATCH --array=0-" + str(njobs-1) + "\n" +\
	"ml anaconda3-2019.03 cuda/11.2.0 cudnn\n" +\
	"conda activate cuda11\n" +\
	"python circuitmap/scripts/run_loho_cv_cavi_sns.py --data %s --out %s --test_hologram_id $SLURM_ARRAY_TASK_ID --reader %s"%(
		data_path + fname, out, reader
	)
	script_filename = "scripts/loho_cv_jobscripts/cavi_sns/loho_cv_jobscript_cavi_sns_%s.sh"%(fname.split('.')[0])
	save_and_call_script(script, script_filename)

	# "#SBATCH --gres=gpu:1\n" +\
	# "#SBATCH --exclude=ax[11-17]\n" +\

def generate_script_cosamp(data_path, fname, njobs, out, reader, caviar_path):
	script = "#!/bin/bash\n" +\
	"#SBATCH --job-name=loho_cv\n" +\
	"#SBATCH -c 1\n" +\
	"#SBATCH -o logs/slurm-%A_%a.out\n" +\
	"#SBATCH --mem-per-cpu=8gb\n" +\
	"#SBATCH --array=0-" + str(njobs-1) + "\n" +\
	"ml anaconda3-2019.03 cuda/11.2.0 cudnn\n" +\
	"conda activate cuda11\n" +\
	"python circuitmap/scripts/run_loho_cv_cosamp.py --data %s --out %s --test_hologram_id $SLURM_ARRAY_TASK_ID --reader %s --caviar_path %s"%(
		data_path + fname, out, reader, caviar_path
	)
	script_filename = "scripts/loho_cv_jobscripts/cosamp/loho_cv_jobscript_cosamp_%s.sh"%(fname.split('.')[0])
	save_and_call_script(script, script_filename)
	return

def save_and_call_script(script, script_filename):
	with open(script_filename, "w") as myfile:
		myfile.write(script)
	myfile.close()
	subprocess.call('sbatch ' + script_filename, shell=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path')
	parser.add_argument('--method')
	parser.add_argument('--reader')
	parser.add_argument('--demixer')
	parser.add_argument('--msrmp')
	parser.add_argument('--out')
	parser.add_argument('--caviar_path')
	args = parser.parse_args()

	files = os.listdir(args.data_path)

	for f in files:
		print('Generating LOHO-CV job-array script for %s'%f)
		# Load data
		if args.reader in ['h5py', 'phorc']:
			data = h5py.File(args.data_path + f)
			stim_matrix = np.array(data['stimulus_matrix']).T

		elif args.reader == 'scipy':
			data = loadmat(args.data_path + f)
			stim_matrix = data['stimulus_matrix']

		else:
			raise Exception

		stim_matrix = stim_matrix.astype(float)
		multi_tar_locs = np.where(np.sum(stim_matrix > 0, axis=0) > 1)[0]
		stim_multi = stim_matrix[:, multi_tar_locs]

		# Subset to appropriate training data
		stim_bin = (stim_multi != 0).astype(float)
		unique_holograms = np.vstack({tuple(row) for row in stim_bin.T})
		n_unique_holograms = len(unique_holograms)

		if args.method == 'caviar':
			generate_script_caviar(args.data_path, f, n_unique_holograms, args.demixer, args.msrmp, args.out, args.reader)
		elif args.method == 'cavi_sns':
			generate_script_cavi_sns(args.data_path, f, n_unique_holograms, args.out, args.reader)
		elif args.method == 'cosamp':
			cav = args.caviar_path + f[:-4] + '_msrmp0.4_models.mat'
			generate_script_cosamp(args.data_path, f, n_unique_holograms, args.out, args.reader, cav)
		else:
			raise Exception