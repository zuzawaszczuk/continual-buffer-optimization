import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

benchmark = get_benchmark(config.dataset)
