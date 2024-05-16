# Rocket feature experiments

The initial data is generated with `python initialize_rocket.py`.

The optimal data is generated with `python get_pareto_rocket.py`.

The experiment scripts are runnable with `python run_rocket_label.py <seed>`.

- `label` is the label of the algorithm that is used. 

- `<seed>` specifies the initial seed for torch and numpy. The script will write its output file in `<dirname>/<label>/<seed>_<label>.pt` where `<seed>` is written with 4 digits with as many zeros filling in as needed.