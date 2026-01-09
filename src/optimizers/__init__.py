from typing import Type

from .genetic_algorithm import GeneticAlgorithm
from .hill_climbing import HillClimbing
from .optimizer import Optimizer
from .pbil import PBILOptimizer
from .random_search import RandomSearch

__all__ = ["HillClimbing", "RandomSearch"]

STRATEGY_MAP: dict[str, Type[Optimizer]] = {
    "RandomSearch": RandomSearch,
    "HillClimbing": HillClimbing,
    "Genetic": GeneticAlgorithm,
    "PBIL": PBILOptimizer,
    # "Heuristic": HeuristicOptimizer
}


def get_optimizer(name: str) -> Type[Optimizer]:
    if name not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown optimizer strategy: {name}. Available: {list(STRATEGY_MAP.keys())}"
        )
    return STRATEGY_MAP[name]
