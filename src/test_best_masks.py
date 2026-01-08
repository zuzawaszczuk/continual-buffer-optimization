import pickle
import yaml
from config import Config
from objective import Function, TestFunction
from objective.manage_benchmark import get_benchmark

folder_name = "SplitMNIST_100_ncall_2026-01-07 10:205"

with open(f"{folder_name}/masks.pkl", 'rb') as f:
    best_mask_per_strategy = pickle.load(f)

with open(f"{folder_name}/history_config.yaml", "r") as file:
    data_config = yaml.safe_load(file)


config = Config(**data_config)
benchmark = get_benchmark(config.dataset)

func = Function(benchmark, config.model)
test_func = TestFunction(benchmark, config.model)


for best_masks in best_mask_per_strategy:
    print(best_mask_per_strategy[best_masks])

