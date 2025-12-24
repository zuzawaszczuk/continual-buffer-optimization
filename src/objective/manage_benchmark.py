import numpy as np
import torch
from avalanche.benchmarks import (
    NCScenario,
    benchmark_with_validation_stream,
    split_validation_class_balanced,
)
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset

from config import DatasetConfig


def get_benchmark(dataset_config: DatasetConfig) -> NCScenario:
    torch.manual_seed(dataset_config.seed)
    np.random.seed(dataset_config.seed)

    full_benchmark = SplitMNIST(
        n_experiences=5,
        seed=dataset_config.seed,
    )

    train_experiences = full_benchmark.train_stream[: dataset_config.n_tasks]
    test_experiences = full_benchmark.test_stream[: dataset_config.n_tasks]

    benchmark = benchmark_from_datasets(
        train=[as_classification_dataset(exp.dataset) for exp in train_experiences],
        test=[as_classification_dataset(exp.dataset) for exp in test_experiences],
    )

    benchmark = benchmark_with_validation_stream(
        benchmark,
        seed=dataset_config.seed,
        split_strategy=lambda dataset: split_validation_class_balanced(
            dataset_config.eval_ratio, dataset
        ),
    )

    return benchmark


def calculate_mask_length(benchmark: NCScenario) -> int:
    return sum(len(exp.dataset) for exp in benchmark.train_stream[:-1])


def print_dataset_stats(benchmark: NCScenario) -> None:
    train_stream = benchmark.train_stream
    val_stream = benchmark.valid_stream
    test_stream = benchmark.test_stream

    print("\n=== Benchmark Stats ===\n")
    for experience in train_stream:
        task_id = experience.current_experience
        classes = experience.classes_in_this_experience
        print(f"Task {task_id} | Classes in this task: {classes}")

        train_dataset = experience.dataset
        print(f"  Training examples: {len(train_dataset)}")

        val_dataset = val_stream[task_id].dataset
        print(f"  Validation examples: {len(val_dataset)}")

        test_dataset = test_stream[task_id].dataset
        print(f"  Test examples: {len(test_dataset)}")

        print("-" * 50)
