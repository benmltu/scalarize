# Scalarize experiments

The experiment script is runnable with `python run_experiments.py <dirname> <label> <seed>`.

- `<dirname>` specifies the location for both the configuration and output files. Specifically, the `run_experiments` script will read its configuration from `<dirname>/config.json`. For our experiments, the specific configurations (`config.json`) are available in the different sub-directories.

- `<label>` is a string that specifies which algorithm to use.

- `<seed>` specifies the initial seed for torch and numpy. The script will write its output file in `<dirname>/<label>/<seed>_<label>.pt` where `<seed>` is written with 4 digits with as many zeros filling in as needed.

## Algorithms
There are many algorithms (`<label>`) in the repository. Below, we briefly list the main approaches.

- `sobol` implements the random search algorithm.
- `eui` implements the expected utility improvement (EUI) acquisition function.
- `eui-rg-#` implements the randomised-greedy variant of the EUI acquisition function with parameter `#`.
- `eui-thresh-#` implements the deterministic threshold variant of the EUI acquisition function with parameter `#`.
- `eui-mc-#` implements the EUI acquisition function with `#` number of scalarization parameter samples.
- `eui-fs-#` implements the EUI acquisition function with `#` number of function samples.
- `eui-ts` implements the EUI acquisition function with a Thompson sample as the surrogate model.
- `eui-ucb` implements the EUI acquisition function with an upper confidence bound as the surrogate model.
- `resi` implements the random expected scalarized improvement (RESI) acquisition function.
- `resi-ts` implements the RESI acquisition function with a Thompson sample as the surrogate model.
- `resi-ucb` implements the RESI acquisition function with an upper confidence bound as the surrogate model.
- `ehvi` implements the expected hypervolume improvement (EHVI) acquisition function.
- `nehvi` implements the noisy EHVI acquisition function.
- `parego` implements the ParEGO algorithm, which is the RESI with the augmented Chebyshev scalarization function.
- `nparego` implements the NParEGO algorithm, which is the RESI with the augmented Chebyshev scalarization function. The main difference with the `parego` approach is that the expectation for NParEGO is also over the earlier observations.
