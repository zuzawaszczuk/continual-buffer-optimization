from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from avalanche.benchmarks import NCScenario
from config import ModelConfig
from objective.function import Function


class Optimizer(ABC):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: Dict[str, Any],
    ):
        self.benchmark = benchmark
        self.model_config = model_config
        self.hyperparams = hyperparams

        self.function = Function(self.benchmark, self.model_config)
        self.task_sizes = {
            i: len(exp.dataset)
            for i, exp in enumerate(self.benchmark.train_stream[:-1])
        }
        self.buffer_size = self.hyperparams.get("buffer_size_per_task", 500)

    @abstractmethod
    def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
        pass

    def evaluate(self, masks: Dict[int, np.ndarray]) -> float:
        return self.function(masks)

    def _create_random_mask(self, total_size: int, n_ones: int) -> np.ndarray:
        indices = np.random.choice(total_size, n_ones, replace=False)
        return indices
