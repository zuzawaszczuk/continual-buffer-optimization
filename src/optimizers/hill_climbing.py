import logging
from typing import Dict, Tuple, TypeAlias

import numpy as np
from avalanche.benchmarks import NCScenario
from tqdm import tqdm

from config import HyperparamStrategyConfig, ModelConfig

from .optimizer import Optimizer

Solution: TypeAlias = Dict[int, np.ndarray]


class HillClimbing(Optimizer):
    def __init__(
        self,
        benchmark: NCScenario,
        model_config: ModelConfig,
        hyperparams: HyperparamStrategyConfig,
        logger: logging.Logger,
    ):
        super().__init__(benchmark, model_config, hyperparams, logger)
        self.restarts = self.params.get("restart", 1)
        self.steps_per_restart = max(1, self.n_calls // self.restarts)

    def optimize(self) -> Tuple[float, Solution]:
        best_global_score = -1.0
        best_global_masks = None

        # print(f"HillClimbing ({self.restarts} restarts)")

        for r in range(self.restarts):
            current_masks = {
                tid: self._create_random_mask(size, self.buffer_size)
                for tid, size in self.task_sizes.items()
            }
            current_score = self.evaluate(current_masks)

            if current_score > best_global_score:
                best_global_score = current_score
                best_global_masks = current_masks.copy()

            pbar = tqdm(range(self.steps_per_restart), desc=f"Restart {r+1}")
            for _ in pbar:
                neighbor_masks = self.get_neighbor(current_masks)
                neighbor_score = self.evaluate(neighbor_masks)

                if neighbor_score >= current_score:
                    current_masks = neighbor_masks
                    current_score = neighbor_score

                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_masks = current_masks.copy()
                    pbar.set_postfix({"best": f"{best_global_score:.4f}"})

        assert best_global_masks is not None
        return best_global_score, best_global_masks

    def get_neighbor(self, masks: Dict[int, np.ndarray]) -> Solution:
        neighbor = {}
        for task_id, mask in masks.items():
            new_mask = mask.copy()
            swaps = max(1, int(0.05 * self.buffer_size))
            total_size = self.task_sizes[task_id]
            all_indices = np.arange(total_size)
            available = np.setdiff1d(all_indices, new_mask)

            if len(available) > 0:
                swap_out = np.random.choice(new_mask, size=swaps, replace=False)
                swap_in = np.random.choice(available, size=swaps, replace=False)

                new_mask[np.isin(new_mask, swap_out)] = swap_in

            neighbor[task_id] = new_mask

        return neighbor
