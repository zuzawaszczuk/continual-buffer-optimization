import logging
import pickle
from typing import List
import pandas as pd
import yaml
import numpy as np

from config import Config
from objective import Function, TestFunction
from objective.manage_benchmark import get_benchmark
from utils import plot_heatmaps

folder_name = "SplitMNIST_100_ncall_2026-01-07 11:35"

with open(f"{folder_name}/masks.pkl", "rb") as f:
    best_mask_per_strategy = pickle.load(f)

with open(f"{folder_name}/history_config.yaml", "r") as file:
    data_config = yaml.safe_load(file)

fh = logging.FileHandler(f"{folder_name}/logs.txt")
fh.setLevel(logging.DEBUG)
logger = logging.getLogger("GA_logger")
logger.setLevel(logging.INFO)
logger.addHandler(fh)

config = Config(**data_config)
benchmark = get_benchmark(config.dataset)

func = Function(benchmark, config.model)
test_func = TestFunction(benchmark, config.model)
metrics: List[pd.DataFrame] = []


best_mask_per_strategy["Without Buffer"] = {
    0: np.array([2]),
    1: np.array([2]),
    2: np.array([2]),
    3: np.array([])
}
for best_masks in best_mask_per_strategy:
    print(best_masks)
    function_value = func(best_mask_per_strategy[best_masks])
    logger.info(f"Function value {best masks} {function_value}")

    test_value = test_func(best_mask_per_strategy[best_masks])
    logger.info(
        f"Test value {best masks} {test_value['all']['Top1_Acc_Stream/eval_phase/test_stream']}"
    )

    prefix = "Top1_Acc_Exp/eval_phase/test_stream"

    df = pd.DataFrame(
        {
            f"after_{key}": [
                v for k, v in sorted(metrics.items())
                if k.startswith(prefix)
            ]
            for key, metrics in test_value.items()
        },
        index=["Exp000", "Exp001", "Exp002", "Exp003", "Exp004"],
    )
    metrics.append(df)


plot_heatmaps(metrics, list(best_mask_per_strategy.keys()), folder_name)
