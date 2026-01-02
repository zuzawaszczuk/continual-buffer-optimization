from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .optimizer import Optimizer


class HillClimbing(Optimizer):
    def __init__(self, benchmark, model_config, hyperparams):
        super().__init__(benchmark, model_config, hyperparams)
        self.n_calls = self.hyperparams.get("n_calls", 50)
        self.restarts = self.hyperparams.get("restart", 1)
        self.steps_per_restart = max(1, self.n_calls // self.restarts)

    def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
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

        return best_global_score, best_global_masks

    def get_neighbor(self, masks: Dict[int, np.ndarray]):
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
