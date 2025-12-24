import yaml

from config import Config
from objective.manage_benchmark import get_benchmark, print_dataset_stats

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

print(config.dataset.name)
benchmark = get_benchmark(config.dataset)
print_dataset_stats(benchmark)
