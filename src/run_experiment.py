import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark
from optimizers import get_optimizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

benchmark = get_benchmark(config.dataset)

for strategy_conf in config.strategies:
    print(f"Running strategy: {strategy_conf.name}")

    OptimizerClass = get_optimizer(strategy_conf.name)

    optimizer = OptimizerClass(
        benchmark=benchmark,
        model_config=config.model,
        hyperparams=strategy_conf.params | {"n_calls": strategy_conf.n_calls},
    )

    best_score, best_masks = optimizer.optimize()

    print(f"Best accuracy for {strategy_conf.name}: {best_score}")
