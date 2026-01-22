import copy
import random
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
from avalanche.benchmarks import NCScenario

from config import HyperparamStrategyConfig, ModelConfig

from .optimizer import Optimizer

Solution: TypeAlias = Dict[int, np.ndarray]


@dataclass
class Individual:
    masks: Solution
    fitness: Optional[float] = None


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
        self.calls_used = 0
        self.population: List[Individual] = [
            self.create_random_individual() for _ in range(self.population_size)
        ]
        self.n_elite = self.params.get("n_elite", 1)
        self.epochs = self.params.get("epochs", 1)

    def optimize(self) -> Tuple[float, Solution]:
        best_global_ind = self.population[0]

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch: {epoch}")

            if self.calls_used >= self.n_calls:
                break

            new_population = self.tournament_selection()
            new_population = self.reproduction(new_population)

            for ind in new_population:
                if self.calls_used >= self.n_calls:
                    break
                    
                if np.random.uniform(0, 1) < self.params.get("mutation_rate", 0.1):
                    ind.masks = self.mutation(ind.masks)
                    ind.fitness = self.evaluate(ind.masks)

            self.population, best_ind = self.elite_selection(new_population)
            
            if best_ind.fitness is not None:
                if best_global_ind.fitness is None or best_ind.fitness > best_global_ind.fitness:
                    best_global_ind = copy.deepcopy(best_ind)

        final_score = best_global_ind.fitness if best_global_ind.fitness is not None else 0.0
        return final_score, best_global_ind.masks

    def tournament_selection(self, k: int = 2) -> List[Individual]:
        new_population = []

        for _ in range(self.population_size):
            contenders = random.sample(self.population, k)

            best = max(
                contenders,
                key=lambda ind: -np.inf if ind.fitness is None else ind.fitness,
            )

            new_population.append(
                Individual(masks=copy.deepcopy(best.masks), fitness=best.fitness)
            )

        return new_population

    def reproduction(self, population: List[Individual]) -> List[Individual]:
        new_population = []
        for i in range(0, len(population) - 1, 2):
            if np.random.uniform(0, 1) < self.params.get("reproduce_rate", 0.1):
                new_pop1, new_pop2 = self.reproduce(
                    population[i].masks, population[i + 1].masks
                )
                new_population.append(Individual(new_pop1, self.evaluate(new_pop1)))
                new_population.append(Individual(new_pop2, self.evaluate(new_pop2)))

        return population + new_population

    def reproduce(
        self, solution_a: Solution, solution_b: Solution
    ) -> Tuple[Solution, Solution]:
        keys = sorted(list(solution_a.keys()))
        if len(keys) == 1:
            tid = keys[0]
            mask1 = solution_a[tid]
            mask2 = solution_b[tid]
            if len(mask1) > 1:
                cut = np.random.randint(1, len(mask1))
                new_mask1 = np.concatenate((mask1[:cut], mask2[cut:]))
                new_mask2 = np.concatenate((mask2[:cut], mask1[cut:]))
                return {tid: new_mask1}, {tid: new_mask2}
            else:
                return solution_a, solution_b
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

        if len(available) > 0:
            idx_to_change = np.random.randint(0, len(masks[idx_mask]))
            masks[idx_mask][idx_to_change] = np.random.choice(available)

        return masks

    def elite_selection(
        self, population: List[Individual]
    ) -> Tuple[List[Individual], Individual]:
        valid_population = [ind for ind in population if ind.fitness is not None]
        
        if not valid_population:
            return population[:self.population_size], population[0]

        sorted_pop = sorted(valid_population, key=lambda ind: ind.fitness, reverse=True)  # type: ignore

        elites = sorted_pop[: self.n_elite]
        best_ind = elites[0]

        remaining = sorted_pop[self.n_elite :]
        k_needed = self.population_size - len(elites)
        if len(remaining) >= k_needed:
            new_population = random.sample(remaining, k=k_needed)
        else:
            new_population = remaining + random.choices(elites, k=k_needed - len(remaining))

        return elites + new_population, best_ind

    def create_random_individual(self) -> Individual:
        masks = self._create_random_solution()
        value = self.evaluate(masks)

        return Individual(masks, value)

    def evaluate(self, masks: Solution) -> float | None:
        if self.calls_used >= self.n_calls:
            return None

        return self._evaluate(masks)
