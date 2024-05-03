# Scalarize

This repository contains the code for our papers:

1. Multi-objective optimisation via the R2 utilities (https://arxiv.org/abs/2305.11774)
2. Random Pareto front surfaces (https://arxiv.org/abs/2405.01404)

At its core, the scalarize code provides a useful collection of utilities for scalarization functions, scalarization parameters and scalarized objective functions that can be used alongside standard routines from NumPy, SciPy and PyTorch.

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
- The notebook folder contains the notebooks used to obtain the plots used our papers.
- The plot_experiments folder contains the notebooks used to plot the results of the experiments.
- The scalarize folder contains the code needed to execute these strategies. 
- The visualize folder contains the streamlit apps that can be used to visualize the one-dimensional or two-dimensional slices of a Pareto front surface.
