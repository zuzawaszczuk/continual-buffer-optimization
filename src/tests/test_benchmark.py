from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.scenarios import NCScenario

from objective import calculate_mask_length


def test_calculate_mask_length_splitmnist() -> None:

    benchmark: NCScenario = SplitMNIST(n_experiences=5, seed=42)
    mask_length = calculate_mask_length(benchmark)

    assert mask_length, 39037
