# Scalarize experiments

The experiment script is runnable with `python run_robust_experiments.py <dirname> <label> <seed>`.

- `<dirname>` specifies the location for both the configuration and output files. Specifically, the `run_robust_experiments` script will read its configuration from `<dirname>/config.json`. For our experiments, the specific configurations (`config.json`) are available in the different sub-directories.

- `<label>` is a string that specifies which algorithm to use.

- `<seed>` specifies the initial seed for torch and numpy. The script will write its output file in `<dirname>/<label>/<seed>_<label>.pt` where `<seed>` is written with 4 digits with as many zeros filling in as needed.

## Algorithms
There are many algorithms (`<label>`) in the repository. Below, we briefly describe the main approaches.

- `sobol` implements the random search algorithm.
- `eui` implements the expected utility improvement (EUI) acquisition function.
- `eui-ts` implements the EUI acquisition function with a Thompson sample as the surrogate model.
- `eui-ucb` implements the EUI acquisition function with an upper confidence bound as the surrogate model.
- `resi` implements the random expected scalarized improvement (RESI) acquisition function.
- `resi-ts` implements the RESI acquisition function with a Thompson sample as the surrogate model.
- `resi-ucb` implements the RESI acquisition function with an upper confidence bound as the surrogate model.
- `aresi-ucb` implements the RESI acquisition function with the mean as the surrogate model and adds an uncertainty penalty.
- `ehvi` implements the expected hypervolume improvement (EHVI) acquisition function.
- `nehvi` implements the noisy EHVI acquisition function.
- `parego` implements the ParEGO algorithm, which is the RESI with the augmented Chebyshev scalarization function.
- `nparego` implements the NParEGO algorithm, which is the RESI with the augmented Chebyshev scalarization function. The main difference with the `parego` approach is that the expectation for NParEGO is also over the earlier observations.
- `robust-eui` implements the robust variant of the `eui` approach.
- `robust-eui-ts` implements the robust variant of the `eui-ts` approach.
- `robust-eui-ucb` implements the robust variant of the `eui-ucb` approach.
- `robust-resi` implements the robust variant of the `resi` approach.
- `robust-resi-ts` implements the robust variant of the `resi-ts` approach.
- `robust-resi-ucb` implements the robust variant of the `resi-ucb` approach.
