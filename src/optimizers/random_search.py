from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from .optimizer import Optimizer


class RandomSearch(Optimizer):
    def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
        best_score = -1.0
        best_masks = None
        n_calls = self.hyperparams.get("n_calls", 10)

        # print(f"RandomSearch ({n_calls} calls)")

        for _ in tqdm(range(n_calls)):
            current_masks = {}
            for task_id, size in self.task_sizes.items():
                current_masks[task_id] = self._create_random_mask(
                    size, self.buffer_size
                )
                # print("Current mask for task", task_id)
                # print(current_masks[task_id])
            score = self.evaluate(current_masks)

            if score > best_score:
                best_score = score
                best_masks = current_masks.copy()
                # print(f"New best random: {best_score}")

        return best_score, best_masks
