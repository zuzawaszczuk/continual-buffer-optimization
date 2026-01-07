from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, List, Tuple, TypeAlias

import numpy as np
from avalanche.benchmarks import NCScenario

from config import HyperparamStrategyConfig, ModelConfig
from objective.function import Function

Solution: TypeAlias = Dict[int, np.ndarray]


class Optimizer(ABC):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: HyperparamStrategyConfig,
        logger: Logger = Logger("default"),
    ):
        self.benchmark = benchmark
        self.model_config = model_config
        self.params = hyperparams.params
        self.logger = logger

        self.function = Function(self.benchmark, self.model_config)
        self.task_sizes = {
            i: len(exp.dataset)
            for i, exp in enumerate(self.benchmark.train_stream[:-1])
        }
        self.buffer_size = hyperparams.buffer_size
        self.n_calls = hyperparams.n_calls
        self.calls_used = 0
        self.history: List[float] = []

    @abstractmethod
    def optimize(self) -> Tuple[float, Solution]:
        pass

    def _evaluate(self, masks: Solution) -> float:
        if self.calls_used >= self.n_calls:
            print("All function calls were already used")

        value = self.function(masks)
        self.history.append(value)
        self.calls_used += 1
        return value

    def _create_random_mask(self, total_size: int, n_ones: int) -> np.ndarray:
        indices = np.random.choice(total_size, n_ones, replace=False)
        return indices

    def _create_random_solution(self) -> Solution:
        solution = {
            i: self._create_random_mask(self.task_sizes[i], self.buffer_size)
            for i, exp in enumerate(self.benchmark.train_stream[:-1])
        }

        return solution
