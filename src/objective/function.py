from typing import Any, Dict

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


class Function:
    def __init__(self, benchmark: NCScenario, model_config: ModelConfig):
        self.benchmark = benchmark
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, masks: Dict[int, np.ndarray]) -> float:
        task_buffers = self.get_tasks_buffers(masks)

        criterion_cls = CRITERION_MAP[self.model_config.criterion]
        optimizer_cls = OPTIMIZER_MAP[self.model_config.optimizer]
        model_cls = MODEL_MAP[self.model_config.type]

        model = model_cls(
            num_classes=self.model_config.output_dim,
            input_size=self.model_config.input_dim,
            hidden_size=self.model_config.hidden_dims,
            hidden_layers=1,
            drop_rate=self.model_config.dropout,
        )

        evaluator = EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            loss_metrics(experience=True, stream=True),
            loggers=[],
        )

        cl_strategy = OptimizedBufferStrategy(
            model=model,
            optimizer=optimizer_cls(
                model.parameters(), lr=self.model_config.lr, momentum=0.9
            ),
            criterion=criterion_cls(),
            task_buffers=task_buffers,
            train_mb_size=self.model_config.batch,
            train_epochs=self.model_config.epochs,
            eval_mb_size=self.model_config.batch,
            device=self.device,
            evaluator=evaluator,
        )

        for experience in self.benchmark.train_stream:
            cl_strategy.train(experience)

        metrics = self.eval(cl_strategy)
        return float(metrics["Top1_Acc_Stream/eval_phase/valid_stream"])

    def eval(self, cl_strategy: OptimizedBufferStrategy) -> Any:
        metrics = cl_strategy.eval(self.benchmark.valid_stream)
        return metrics

    def get_tasks_buffers(
        self, masks: Dict[int, np.ndarray]
    ) -> Dict[int, AvalancheDataset]:
        task_buffers = {}
        cumulative_buffer = None

        for task_id, exp in enumerate(self.benchmark.train_stream[:-1]):
            current_buffer = exp.dataset.subset(masks[task_id])

            if cumulative_buffer:
                cumulative_buffer = AvalancheDataset.concat(
                    cumulative_buffer, current_buffer
                )
            else:
                cumulative_buffer = current_buffer

            task_buffers[task_id] = cumulative_buffer

        return task_buffers
