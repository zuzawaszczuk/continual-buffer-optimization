from typing import Type

from .hill_climbing import HillClimbing
from .optimizer import Optimizer
from .random_search import RandomSearch

__all__ = ["HillClimbing", "RandomSearch"]

STRATEGY_MAP = {
    "RandomSearch": RandomSearch,
    # "HillClimbing": HillClimbing,
    # "Genetic": GeneticOptimizer,
    # "PBIL": PBILOptimizer,
    # "Heuristic": HeuristicOptimizer
}


def get_optimizer(name: str) -> Type[Optimizer]:
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown optimizer strategy: {name}. Available: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name]
