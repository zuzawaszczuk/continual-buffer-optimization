import numpy as np
import pytest
import yaml

from config import Config
from objective.manage_benchmark import get_benchmark
from optimizers import RandomSearch


@pytest.fixture
def test_optimizer() -> RandomSearch:
    with open("src/tests/test_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    config = Config(**data)
    benchmark = get_benchmark(config.dataset)
    return RandomSearch(benchmark, config.model, config.strategies[0])


def test_create_random_solution(test_optimizer: RandomSearch) -> None:
    solution = test_optimizer._create_random_solution()

    assert isinstance(solution, dict)

    assert len(solution) == 4

    for task_id, mask in solution.items():
        assert isinstance(mask, np.ndarray)

        assert len(mask) == 100

        assert len(mask) == len(set(mask))

        assert mask.min() >= 0
        assert mask.max() < test_optimizer.task_sizes[task_id]
