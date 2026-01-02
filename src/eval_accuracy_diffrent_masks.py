import numpy as np
import torch
import yaml

from config import Config
from objective import Function
from objective.manage_benchmark import get_benchmark, print_dataset_stats

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

print(config.dataset.name)
benchmark = get_benchmark(config.dataset)
print_dataset_stats(benchmark)

masks = {}
for i, task in enumerate(benchmark.train_stream[:-1]):
    masks[i] = np.arange(len(task.dataset))

f = Function(benchmark, config.model)
acc = f(masks)

print(f"Accuracy on eval dataset, with full buffer {acc}")

for i, task in enumerate(benchmark.train_stream[:-1]):
    masks[i] = np.random.choice(np.arange(0, len(task.dataset)), size=10)

f = Function(benchmark, config.model)
acc = f(masks)

print(f"Accuracy on eval dataset, with 10 samples per task added to buffer {acc}")
