from __future__ import annotations

from typing import Optional, Union, Dict
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from avalanche.benchmarks.scenarios import Experience
from avalanche.training.templates.strategy_mixin_protocol import CriterionType


class OptimizedBufferStrategy:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        task_buffers: Optional[Dict[int, ParametricBuffer]] = None,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
        )
        self.task_buffers = task_buffers

    def train_dataset_adaptation(self, dataset: AvalancheDataset) -> AvalancheDataset:
        task_id = self.experience.task_label
        
        if task_id in self.task_buffers:
            buffer_dataset = self.task_buffers[task_id].to_dataset()
            dataset = AvalancheConcatDataset([buffer_dataset, dataset])
        
        return dataset
    
    def train(self, experience: Experience) -> None:
        super().train(experience)

    def eval(self, experience: Experience) -> None:
        pass
