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
├── tests/
│   └── test_benchmark.py  
├── config.py                 # Pydantic structure of config
└── run_experiment.py         # Script to run experiments
```

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