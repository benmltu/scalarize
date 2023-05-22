# Scalarize

This repository contains the code for the scalarization-based acquisition functions described in our paper (https://arxiv.org/abs/2305.11774). All of the strategies are implemented based on the BoTorch library (https://github.com/pytorch/botorch/). 

### Dependencies
This code was initially implemented with the following dependencies:

- Python 3.9
- BoTorch 0.8.1
- PyTorch 1.13.1
- GPytorch 1.9.1
- Linear-operator 0.3.0
- SciPy 1.9.3
- NumPy 1.23.5

### Organization

- The experiments folder contains the scripts and configurations which are used to execute the experiments. 
- The notebook folder contains the notebooks used to obtain the plots used in the paper.
- The plot_experiments folder contains the notebooks used to plot the results of the experiments.
- The scalarize folder contains the code needed to execute these strategies. The folder is organised is a similar way to the BoTorch repository.

If you want to discuss more about the code presented here, feel free to e-mail me ben.tu16@imperial.ac.uk.
