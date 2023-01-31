# Scalarize experiments

The experiment script follows a similar pattern to the one in the robust_mobo repository (https://github.com/facebookresearch/robust_mobo/tree/main/experiments).

The experiment script is runnable with `python run_experiments.py <dirname> <label> <seed>`.

- `<dirname>` specifies the location for both the configuration and output files. Specifically, the `run_experiments` script will read its configuration from `<dirname>/config.json`. For our experiments, the specific configurations (`config.json`) are available in the different sub-directories.

- `<label>` is a string that specifies which algorithm to use.

- `<seed>` specifies the initial seed for torch and numpy. The script will write its output file in `<dirname>/<label>/<seed>_<label>.pt` where `<seed>` is written with 4 digits with as many zeros filling in as needed.
