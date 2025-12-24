from avalanche.benchmarks.classic import SplitMNIST

from objective import get_tasks_buffers
import numpy as np


def test_get_tasks_buffers():
    benchmark = SplitMNIST(n_experiences=5, return_task_id=True)

    total_samples = sum(len(exp.dataset) for exp in benchmark.train_stream)
    mask = np.zeros(total_samples, dtype=int)
    mask[::2] = 1

    task_buffers = get_tasks_buffers(benchmark, mask)

    assert len(task_buffers) == len(benchmark.train_stream) - 1

    # # Sprawdzamy, że wszystkie przykłady w bufory mają mask=1
    # mask_idx = 0
    # for exp in benchmark.train_stream[:-1]:
    #     buf = task_buffers[exp.id + 1]
    #     num_samples = len(exp.dataset)
    #     current_mask = mask[mask_idx : mask_idx + num_samples]
    #     mask_idx += num_samples

    #     if buf is not None:
    #         # Wszystkie indeksy w buf powinny odpowiadać pozycji mask=1
    #         buf_indices = buf.indices
    #         for idx in buf_indices:
    #             assert current_mask[idx] == 1

    # print("Test passed: get_tasks_buffers działa poprawnie.")