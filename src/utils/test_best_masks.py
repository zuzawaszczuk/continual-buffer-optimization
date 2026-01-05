import numpy as np

loaded = np.load("SplitMNIST_2026-01-05 23:58:58_masks.npz", allow_pickle=True)
best_mask_per_strategy = {k: loaded[k] for k in loaded}

print(best_mask_per_strategy)
