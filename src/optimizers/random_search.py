from typing import Dict, Tuple, TypeAlias

import numpy as np
from tqdm import tqdm

from .optimizer import Optimizer

Solution: TypeAlias = Dict[int, np.ndarray]


class RandomSearch(Optimizer):
    def optimize(self) -> Tuple[float, Solution]:
        best_score = -1.0
        best_masks = None
        self.calls_used = 0

        for _ in tqdm(range(self.n_calls)):
            current_masks = self._create_random_solution()
            score = self._evaluate(current_masks)

            if score > best_score:
                best_score = score
                best_masks = current_masks.copy()
                # print(f"New best random: {best_score}")

        assert best_masks is not None
        return best_score, best_masks
