from typing import Any, Dict, Optional, Union

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from torch.nn import Module
from torch.optim import Optimizer


class OptimizedBufferStrategy(SupervisedTemplate):  # type: ignore[misc]
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        task_buffers: Optional[Dict[int, AvalancheDataset]] = None,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        evaluator: EvaluationPlugin,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            evaluator=evaluator,
        )
        self.task_buffers = task_buffers

    def train_dataset_adaptation(self, **kwargs: Any) -> None:
        task_id = (
            self.experience.current_experience - 1
        )  # we take buffer with samples of previos experiments
        dataset = self.experience.dataset

        if self.task_buffers is not None and task_id in self.task_buffers:
            dataset = AvalancheDataset.concat(dataset, self.task_buffers[task_id])

        self.adapted_dataset = dataset
