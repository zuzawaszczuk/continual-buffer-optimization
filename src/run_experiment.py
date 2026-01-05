import datetime
import logging

import pandas as pd
import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark, print_dataset_stats
from optimizers import get_optimizer
from utils import plot_ecdf

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = Config(**data)

start_time = datetime.datetime.now()
file_name = f"{config.dataset.name}_{start_time.strftime('%Y-%m-%d %H:%M:%S')}.txt"

fh = logging.FileHandler(f"{file_name}.txt")
fh.setLevel(logging.DEBUG)
logger = logging.getLogger("GA_logger")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info(config.model_dump_json(indent=4))
benchmark = get_benchmark(config.dataset)
print_dataset_stats(benchmark)

history_ecdf = pd.DataFrame()

for strategy_conf in config.strategies:
    start_time = datetime.datetime.now()
    logger.info(
        f"Running strategy: {strategy_conf.name} | Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    OptimizerClass = get_optimizer(strategy_conf.name)

    optimizer = OptimizerClass(
        benchmark=benchmark,
        model_config=config.model,
        hyperparams=strategy_conf,
        logger=logger,
    )
    history_ecdf[f"{strategy_conf.name}"] = optimizer.history

    best_score, best_masks = optimizer.optimize()

    logger.info(f"Best accuracy for {strategy_conf.name}: {best_score}")
    logger.info(f"Best solution for {best_masks}")

plot_ecdf(history_ecdf, file_name)
