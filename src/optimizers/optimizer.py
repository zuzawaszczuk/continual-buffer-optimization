from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from avalanche.benchmarks import NCScenario

from objective.function import Function
from utils.config import HyperparamStrategyConfig, ModelConfig


class Optimizer(ABC):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: HyperparamStrategyConfig,
    ):
        self.benchmark = benchmark
        self.model_config = model_config

        self.function = Function(self.benchmark, self.model_config)
        self.task_sizes = {
            i: len(exp.dataset)
            for i, exp in enumerate(self.benchmark.train_stream[:-1])
        }
        self.buffer_size = hyperparams.buffer_size
        self.n_calls = hyperparams.n_calls

    @abstractmethod
    def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
        pass

    def evaluate(self, masks: Dict[int, np.ndarray]) -> float:
        return self.function(masks)

    def _create_random_mask(self, total_size: int, n_ones: int) -> np.ndarray:
        indices = np.random.choice(total_size, n_ones, replace=False)
        return indices

    def _create_random_solution(self) -> Dict[int, np.ndarray]:
        solution = {
            i: self._create_random_mask(self.task_sizes[i], self.buffer_size)
            for i, exp in enumerate(self.benchmark.train_stream[:-1])
        }

        return solution
