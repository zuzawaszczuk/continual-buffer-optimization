from typing import Any, Dict

from avalanche.benchmarks import NCScenario

from config import ModelConfig

from .function import Function
from .strategy import OptimizedBufferStrategy


class TestFunction(Function):
    def __init__(self, benchmark: NCScenario, model_config: ModelConfig):
        super().__init__(benchmark, model_config)
        self.metrics: Dict[str, Dict[str, float]] = {}

    def train(self, cl_strategy: OptimizedBufferStrategy) -> None:
        for experience in self.benchmark.train_stream:
            task_id = experience.current_experience
            cl_strategy.train(experience)
            self.metrics[f"{task_id}"] = cl_strategy.eval(self.benchmark.test_stream)

    def eval(self, cl_strategy: OptimizedBufferStrategy) -> Any:
        self.metrics["all"] = cl_strategy.eval(self.benchmark.test_stream)
        return self.metrics
