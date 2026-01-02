import pytest
import yaml

from objective import Function
from objective.manage_benchmark import get_benchmark
from utils import Config


@pytest.fixture
def test_fuction() -> Function:
    with open("src/tests/test_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    config = Config(**data)
    benchmark = get_benchmark(config.dataset)
    return Function(benchmark, config.model)
