import numpy as np
import pytest
from avalanche.benchmarks.classic import SplitCIFAR10

from objective import get_tasks_buffers


def test_get_tasks_buffers_cifar() -> None:
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=True)
    # cifar 10k in train date per 1 task

    masks = {}
    masks[0] = np.random.choice(np.arange(0, 10000), size=100)
    masks[1] = np.random.choice(np.arange(0, 10000), size=100)
    masks[2] = np.random.choice(np.arange(0, 10000), size=100)
    masks[3] = np.random.choice(np.arange(0, 10000), size=100)

    task_buffers = get_tasks_buffers(benchmark, masks)

    assert len(task_buffers) == len(benchmark.train_stream) - 1
    assert len(task_buffers[0]) == 100
    assert len(task_buffers[1]) == 200
    assert len(task_buffers[2]) == 300
    assert len(task_buffers[3]) == 400


def test_get_tasks_buffers_cifar_whole() -> None:
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=True)

    masks = {}
    masks[0] = np.arange(10000)
    masks[1] = np.arange(10000)
    masks[2] = np.arange(10000)
    masks[3] = np.arange(10000)

    task_buffers = get_tasks_buffers(benchmark, masks)

    assert len(task_buffers) == len(benchmark.train_stream) - 1
    assert len(task_buffers[0]) == 10000
    assert len(task_buffers[1]) == 20000
    assert len(task_buffers[2]) == 30000
    assert len(task_buffers[3]) == 40000


def test_get_tasks_buffers_cifar_out_of_index() -> None:
    benchmark = SplitCIFAR10(n_experiences=5, return_task_id=True)

    masks = {}
    masks[0] = np.arange(10001)
    masks[1] = np.arange(10000)
    masks[2] = np.arange(10000)
    masks[3] = np.arange(10000)

    with pytest.raises(IndexError):
        task_buffers = get_tasks_buffers(benchmark, masks)  # noqa: F841
