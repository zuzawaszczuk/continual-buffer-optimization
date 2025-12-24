import numpy as np
import torch
from typing import Dict
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import SimpleMLP
from .strategy import OptimizedBufferStrategy
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam

from config import ModelConfig

CRITERION_MAP = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MSELoss": MSELoss,
}

OPTIMIZER_MAP = {
    "SGD": SGD,
    "Adam": Adam,
}

MODEL_MAP = {
    "SimpleMLP": SimpleMLP,
}


def function(benchmark: NCScenario, model_config: ModelConfig, mask: np.ndarray) -> float:
    task_buffers = get_tasks_buffers(benchmark, mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_cls = CRITERION_MAP[model_config.criterion]
    optimizer_cls = OPTIMIZER_MAP[model_config.optimizer]
    model_cls = MODEL_MAP[model_config.type]

    model = model_cls(num_classes=model_config.output_dim)

    cl_strategy = OptimizedBufferStrategy(
        model=model,
        optimizer=optimizer_cls(model.parameters(), lr=model_config.lr, momentum=0.9),
        criterion=criterion_cls(),
        task_buffers=task_buffers,
        train_mb_size=model_config.batch,
        train_epochs=model_config.epochs,
        eval_mb_size=model_config.batch,
        device=device,
    )


def get_tasks_buffers(benchmark: NCScenario, mask: np.ndarray) -> Dict[int, AvalancheDataset]:
    task_buffers = {}
    mask_idx = 0
    for exp in benchmark.train_stream[:-1]:
        num_samples = len(exp.dataset)
        current_mask = torch.tensor(mask[mask_idx : mask_idx + num_samples])
        mask_idx += num_samples
        selected_indices = torch.nonzero(current_mask).flatten().tolist()

        if selected_indices:
            selected_dataset = AvalancheDataset(exp.dataset, indices=selected_indices)
            task_buffers[exp.id + 1] = selected_dataset

    return task_buffers
