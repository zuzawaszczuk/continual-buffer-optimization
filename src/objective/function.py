from typing import Dict

import numpy as np
import torch
from avalanche.benchmarks import NCScenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam

from config import ModelConfig

from .strategy import OptimizedBufferStrategy

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


def function(
    benchmark: NCScenario, model_config: ModelConfig, masks: Dict[int, np.ndarray]
) -> float:
    task_buffers = get_tasks_buffers(benchmark, masks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_cls = CRITERION_MAP[model_config.criterion]
    optimizer_cls = OPTIMIZER_MAP[model_config.optimizer]
    model_cls = MODEL_MAP[model_config.type]

    model = model_cls(
        num_classes=model_config.output_dim,
        input_size=model_config.input_dim,
        hidden_size=model_config.hidden_dims,
        hidden_layers=1,
        drop_rate=model_config.dropout,
    )

    evaluator = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        loggers=[],
    )

    cl_strategy = OptimizedBufferStrategy(
        model=model,
        optimizer=optimizer_cls(model.parameters(), lr=model_config.lr, momentum=0.9),
        criterion=criterion_cls(),
        task_buffers=task_buffers,
        train_mb_size=model_config.batch,
        train_epochs=model_config.epochs,
        eval_mb_size=model_config.batch,
        device=device,
        evaluator=evaluator,
    )

    for experience in benchmark.train_stream:
        cl_strategy.train(experience)

    metrics = cl_strategy.eval(benchmark.valid_stream)
    return float(metrics["Top1_Acc_Stream/eval_phase/valid_stream"])


def get_tasks_buffers(
    benchmark: NCScenario, masks: Dict[int, np.ndarray]
) -> Dict[int, AvalancheDataset]:
    task_buffers = {}
    cumulative_buffer = None

    for task_id, exp in enumerate(benchmark.train_stream[:-1]):
        current_buffer = exp.dataset.subset(masks[task_id])

        if cumulative_buffer:
            cumulative_buffer = AvalancheDataset.concat(
                cumulative_buffer, current_buffer
            )
        else:
            cumulative_buffer = current_buffer

        task_buffers[task_id] = cumulative_buffer

    return task_buffers
