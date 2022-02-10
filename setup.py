from setuptools import setup

setup(
    name='adaprobe',
    version='0.1.0',    
    description='Circuit mapping with 2p opto-stim',
    url='',
    author='mtriplett',
    author_email='',
    license='BSD 2-clause',
    packages=['adaprobe'],
    install_requires=[                     
	'numpy==1.21',
	'scikit-learn==0.24.2',
	'torch',
	'prettytable',
	'scipy',
	'pytorch-lightning',
	'matplotlib',
	'pandas',
	'PyYAML'
	],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
