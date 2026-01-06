import random
from logging import Logger
from typing import Dict, List, Tuple, TypeAlias

import numpy as np
from avalanche.benchmarks import NCScenario

from config import HyperparamStrategyConfig, ModelConfig

from .optimizer import Optimizer

Solution: TypeAlias = Dict[int, np.ndarray]


class GeneticAlgorithm(Optimizer):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: HyperparamStrategyConfig,
        logger: Logger = Logger("default"),
    ):
        super().__init__(benchmark, model_config, hyperparams, logger)
        self.population_size = self.params.get("population", 20)
        self.population: List[Solution] = [
            self._create_random_solution() for _ in range(self.population_size)
        ]
        self.function_cache: list[float | None] = [None] * self.population_size
        self.n_elite = self.params.get("n_elite", 1)
        self.epochs = self.params.get("epochs", 1)
        self.calls_used = 0

    def optimize(self) -> Tuple[float, Solution]:
        self.calculate()
        best_idx = 0

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch: {epoch}")

            if self.calls_used >= self.n_calls:
                break

            elite_indices = np.argsort(self.function_cache)[-self.n_elite :]  # type: ignore[arg-type]
            elites = [self.population[i] for i in elite_indices]
            elite_cache = [self.function_cache[i] for i in elite_indices]

            self.population, self.function_cache = self.tournament_selection(
                self.params.get("tournament_size", 2)
            )

            for i in range(0, len(self.population) - 1, 2):
                if np.random.uniform(0, 1) > self.params.get("reproduce_rate", 0.1):
                    self.population[i], self.population[i + 1] = self.reproduce(
                        self.population[i], self.population[i + 1]
                    )
                    self.function_cache[i], self.function_cache[i + 1] = None, None

            for i in range(len(self.population)):
                if np.random.uniform(0, 1) > self.params.get("mutation_rate", 0.1):
                    self.population[i] = self.mutation(self.population[i])
                    self.function_cache[i] = None

            self.calculate()
            self.elite_selection(elites, elite_cache)

            values = np.array(
                [-np.inf if v is None else v for v in self.function_cache]
            )

            best_idx = int(np.argmax(values))

        value = self.function_cache[best_idx]
        assert value is not None

        return value, self.population[best_idx]

    def calculate(self) -> None:
        for i, individual in enumerate(self.population):
            if self.function_cache[i] is None and self.calls_used < self.n_calls:
                self.calls_used += 1
                self.function_cache[i] = self.evaluate(individual)

    def tournament_selection(
        self, k: int = 2
    ) -> Tuple[List[Solution], List[float | None]]:
        new_cache: List[None | float] = [None] * self.population_size
        new_population = []

        for i in range(self.population_size):
            tournament_indices = np.random.choice(
                self.population_size, k, replace=False
            )
            best_idx = max(tournament_indices, key=lambda idx: self.function_cache[idx])
            new_population.append(self.population[best_idx])
            new_cache[i] = self.function_cache[best_idx]

        return new_population, new_cache

    def reproduce(
        self, solution_a: Solution, solution_b: Solution
    ) -> Tuple[Solution, Solution]:
        keys = list(solution_a.keys())
        cut = np.random.randint(1, len(keys))

        child_a: Solution = {}
        child_b: Solution = {}

        for k in keys[:cut]:
            child_a[k] = solution_a[k]
            child_b[k] = solution_b[k]

        for k in keys[cut:]:
            child_a[k] = solution_b[k]
            child_b[k] = solution_a[k]

        return child_a, child_b

    def mutation(self, masks: Solution) -> Solution:
        idx_mask = np.random.choice(list(masks.keys()))

        all_indices = np.arange(self.task_sizes[idx_mask])
        available = np.setdiff1d(all_indices, masks[idx_mask])

        idx_to_change = np.random.randint(0, len(masks[idx_mask]) - 1)

        masks[idx_mask][idx_to_change] = np.random.choice(available)

        return masks

    def elite_selection(self, elites: List[Solution], elite_cache: List[float]) -> None:
        indices = random.sample(range(self.population_size), len(elites))

        for elite, cache, idx in zip(elites, elite_cache, indices):
            self.population[idx] = elite
            self.function_cache[idx] = cache
