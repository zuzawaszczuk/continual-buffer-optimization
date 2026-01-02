from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .optimizer import Optimizer


class RandomSearch(Optimizer):
    def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
        best_score = -1.0
        best_masks = None

        for _ in tqdm(range(self.n_calls)):
            current_masks = self._create_random_solution()
            score = self.evaluate(current_masks)

            if score > best_score:
                best_score = score
                best_masks = current_masks.copy()
                # print(f"New best random: {best_score}")

        return best_score, best_masks
