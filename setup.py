from setuptools import setup, find_packages

setup(
	name='circuitmap',
	version='0.2.0',
	description='Neural waveform demixing and synaptic connectivity inference for holographic ensemble stimulation',
	author='Marcus Triplett',
	author_email='marcus.triplett@columbia.edu',
	packages=find_packages(),
	install_requires=[
		'numpy==1.21',
		'scipy==1.6',
		'scikit-learn==0.24',
		'pandas',
		'tqdm',
		'torch',
		'pytorch-lightning',
	],
)