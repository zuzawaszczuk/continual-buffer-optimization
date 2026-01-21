from logging import Logger
from typing import Dict, Tuple, TypeAlias

import numpy as np
from avalanche.benchmarks import NCScenario

from config import HyperparamStrategyConfig, ModelConfig

from .optimizer import Optimizer

Solution: TypeAlias = Dict[int, np.ndarray]


class PBILOptimizer(Optimizer):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: HyperparamStrategyConfig,
        logger: Logger = Logger("default"),
    ):
        super().__init__(benchmark, model_config, hyperparams, logger)

        self.pop_size = self.params.get("population", 20)
        self.learning_rate = self.params.get("learning_rate", 0.1)
        self.mutation_prob = self.params.get("mutation_prob", 0.05)
        self.mutation_shift = self.params.get("mutation_shift", 0.05)
        self.calls_used = 0
        estimated_epochs = (self.n_calls // self.pop_size) + 2
        self.epochs = max(self.params.get("epochs", 1), estimated_epochs)

        self.prob_vectors = {
            tid: np.full(size, 0.5) for tid, size in self.task_sizes.items()
        }

    def optimize(self) -> Tuple[float, Solution]:
        best_global_score = -np.inf
        best_global_solution: Solution = {}
        if self.prob_vectors:
            best_global_solution = self.create_sample()

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch: {epoch}")

            if self.calls_used >= self.n_calls:
                break

            population_scores = []
            population_solutions = []

            for _ in range(self.pop_size):
                if self.calls_used >= self.n_calls:
                    break
                sample = self.create_sample()
                score = self._evaluate(sample)
                population_scores.append(score)
                population_solutions.append(sample)

            if not population_scores:
                break

            best_idx_in_gen = np.argmax(population_scores)
            current_best_score = population_scores[best_idx_in_gen]
            current_best_solution = population_solutions[best_idx_in_gen]

            if current_best_score > best_global_score:
                best_global_score = current_best_score
                best_global_solution = current_best_solution

            self.update_probs(current_best_solution)
            self.mutate_probs()
        return best_global_score, best_global_solution

    def create_sample(self) -> Solution:
        solution: Solution = {}
        for tid, probs in self.prob_vectors.items():
            p_sum = probs.sum()
            if p_sum > 0:
                p_norm = probs / p_sum
            else:
                p_norm = np.ones_like(probs) / len(probs)

            sample_size = min(len(probs), self.buffer_size)            
            selected_indices = np.random.choice(
                len(probs), size=sample_size, replace=False, p=p_norm
            )
            solution[tid] = selected_indices
        return solution

    def update_probs(self, best_solution: Solution) -> None:
        for tid, selected_indices in best_solution.items():
            target_vector = np.zeros_like(self.prob_vectors[tid])
            target_vector[selected_indices] = 1.0

            self.prob_vectors[tid] = (
                self.prob_vectors[tid] * (1.0 - self.learning_rate)
                + target_vector * self.learning_rate
            )

    def mutate_probs(self) -> None:
        for tid in self.prob_vectors:
            mutation_mask = (
                np.random.rand(len(self.prob_vectors[tid])) < self.mutation_prob
            )

            changes = np.random.choice(
                [-self.mutation_shift, self.mutation_shift],
                size=len(self.prob_vectors[tid]),
            )
            self.prob_vectors[tid][mutation_mask] += changes[mutation_mask]
            self.prob_vectors[tid] = np.clip(self.prob_vectors[tid], 0.01, 0.99)