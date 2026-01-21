import datetime
import logging
import os
import pickle

import pandas as pd
import torch
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark, print_dataset_stats
from optimizers import get_optimizer
from utils import plot_ecdf, plot_history

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as file:
    data_config = yaml.safe_load(file)

config = Config(**data_config)

start_time = datetime.datetime.now()
folder_name = f"{config.dataset.name}_100_ncall_{start_time.strftime('%Y-%m-%d %H:%M')}"

os.makedirs(folder_name, exist_ok=True)

with open(f"{folder_name}/history_config.yaml", 'w') as f:
    yaml.dump(data_config, f)

fh = logging.FileHandler(f"{folder_name}/logs.txt")
fh.setLevel(logging.DEBUG)
logger = logging.getLogger("GA_logger")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info(config.model_dump_json(indent=4))
benchmark = get_benchmark(config.dataset)
print_dataset_stats(benchmark)

all_histories = {}
best_mask_per_strategy = {}

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

    best_score, best_masks = optimizer.optimize()

    print(f"{strategy_conf.name} calls {optimizer.calls_used}")

    target_len = strategy_conf.n_calls
    hist = optimizer.history
    if len(hist) > target_len:
        hist = hist[:target_len]
    elif len(hist) < target_len:
        hist.extend([hist[-1] if hist else 0.0] * (target_len - len(hist)))

    all_histories[f"{strategy_conf.name}"] = hist
    best_mask_per_strategy[f"{strategy_conf.name}"] = best_masks

    logger.info(f"Best accuracy for {strategy_conf.name}: {best_score}")

    checkpoint_df = pd.DataFrame({k: pd.Series(v) for k, v in all_histories.items()})
    checkpoint_df.to_csv(f"{folder_name}/history_checkpoint.csv", index=False)

    with open(f"{folder_name}/masks_checkpoint.pkl", 'wb') as f:
        pickle.dump(best_mask_per_strategy, f)
    
    plot_ecdf(checkpoint_df, folder_name)
    plot_history(checkpoint_df, folder_name)

final_df = pd.DataFrame({k: pd.Series(v) for k, v in all_histories.items()})
print(final_df)