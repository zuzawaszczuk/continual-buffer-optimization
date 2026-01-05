from typing import Any
from .function import Function
from .strategy import OptimizedBufferStrategy


class TestFunction(Function):
    def eval(self, cl_strategy: OptimizedBufferStrategy) -> Any:
        metrics = cl_strategy.eval(self.benchmark.test_stream)
        return metrics
