import logging

import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark
from optimizers import get_optimizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

fh = logging.FileHandler("results.txt")
fh.setLevel(logging.DEBUG)
logger = logging.getLogger("GA_logger")
logger.setLevel(logging.INFO)
logger.addHandler(fh)


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
        hyperparams=strategy_conf,
        logger=logger,
    )

    best_score, best_masks = optimizer.optimize()

    print(f"Best accuracy for {strategy_conf.name}: {best_score}")
