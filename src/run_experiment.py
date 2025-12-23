import yaml

from config import Config
from objective.function import objective_function

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

print(config.dataset.name)
objective_function(config)
