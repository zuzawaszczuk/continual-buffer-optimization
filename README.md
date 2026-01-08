# continual_buffer_optimization

## Repo Structure

```
src/
├── objective/
│   ├── __init__.py
│   ├── function.py           # Function class for evaluation
│   ├── manage_benchmark.py   # Functions to manage custom datasets
│   └── strategy.py           # CL strategy with custom replay buffer
├── optimizers/
│   ├── __init__.py
│   ├── optimizer.py          # Base class
│   ├── random_search.py      # RandomSearch class
│   └── hill_climbing.py      # HillClimbing class
│   └── pbil.py               # PBILOptimizer class
├── tests/
│   └── test_benchmark.py  
├── utils/
│   ├── create_ecdf.py        # Create plot image with ECDF curve for each algorithm
│   ├── showcase_function.py  # Shows how diffrent function value is for diffrent buffors
│   └── test_best_masks.py    # TODO: compare best masks from optimizers with heuristic one on test dataset
├── config.py                 # Pydantic structure of config
└── run_experiment.py         # Script to run experiments
```
src/utils/src/utils/eval_accuracy_diffrent_masks.py

## Linting code

```
chmod +x lint.sh
./lint.sh 
```

## Running tests

`
PYTHONPATH=src pytest src/tests/
`

## Running getting benchmark and printing stats
`
python3 src/run_experiment.py 
`

## Test objective function with continual learning for two example masks