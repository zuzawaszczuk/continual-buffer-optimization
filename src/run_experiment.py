import logging

import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark, print_dataset_stats
from optimizers import get_optimizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

fh = logging.FileHandler(f"{config.dataset.name}_results.txt")
fh.setLevel(logging.DEBUG)
logger = logging.getLogger("GA_logger")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

benchmark = get_benchmark(config.dataset)
print_dataset_stats(benchmark)

for strategy_conf in config.strategies:
    logger.info(f"Running strategy: {strategy_conf.name}")

    OptimizerClass = get_optimizer(strategy_conf.name)

    optimizer = OptimizerClass(
        benchmark=benchmark,
        model_config=config.model,
        hyperparams=strategy_conf,
        logger=logger,
    )

    best_score, best_masks = optimizer.optimize()

    logger.info(f"Best accuracy for {strategy_conf.name}: {best_score}")
