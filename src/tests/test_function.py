import numpy as np
import pytest
import yaml

from config import Config
from objective import Function
from objective.manage_benchmark import get_benchmark


@pytest.fixture
def test_function() -> Function:
    with open("src/tests/test_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    config = Config(**data)
    benchmark = get_benchmark(config.dataset)
    return Function(benchmark, config.model)


def test_get_tasks_buffers_cifar(test_function: Function) -> None:

    masks = {}
    masks[0] = np.random.choice(np.arange(0, 10000), size=100)
    masks[1] = np.random.choice(np.arange(0, 10000), size=100)
    masks[2] = np.random.choice(np.arange(0, 10000), size=100)
    masks[3] = np.random.choice(np.arange(0, 10000), size=100)

    task_buffers = test_function.get_tasks_buffers(masks)

    assert len(task_buffers) == len(test_function.benchmark.train_stream) - 1
    assert len(task_buffers[0]) == 100
    assert len(task_buffers[1]) == 200
    assert len(task_buffers[2]) == 300
    assert len(task_buffers[3]) == 400


def test_get_tasks_buffers_cifar_whole(test_function: Function) -> None:
    masks = {}
    masks[0] = np.arange(10000)
    masks[1] = np.arange(10000)
    masks[2] = np.arange(10000)
    masks[3] = np.arange(10000)

    task_buffers = test_function.get_tasks_buffers(masks)

    assert len(task_buffers) == len(test_function.benchmark.train_stream) - 1
    assert len(task_buffers[0]) == 10000
    assert len(task_buffers[1]) == 20000
    assert len(task_buffers[2]) == 30000
    assert len(task_buffers[3]) == 40000


def test_get_tasks_buffers_cifar_out_of_index(test_function: Function) -> None:
    masks = {}
    masks[0] = np.arange(10001)
    masks[1] = np.arange(10000)
    masks[2] = np.arange(10000)
    masks[3] = np.arange(10000)

    with pytest.raises(IndexError):
        task_buffers = test_function.get_tasks_buffers(masks)  # noqa: F841
