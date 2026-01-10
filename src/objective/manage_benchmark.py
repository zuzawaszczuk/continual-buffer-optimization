import logging
from logging import Logger

from avalanche.benchmarks import (
    NCScenario,
    benchmark_with_validation_stream,
    split_validation_class_balanced,
)
from avalanche.benchmarks.classic import (
    SplitCIFAR10,
    SplitCIFAR100,
    SplitCIFAR110,
    SplitMNIST,
)
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset

from config import DatasetConfig


def get_benchmark(dataset_config: DatasetConfig) -> NCScenario:
    full_benchmark = factory_benchmark(dataset_config.name, dataset_config.seed)

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


def print_dataset_stats(benchmark: NCScenario, logger: Logger | None = None) -> None:
    if logger is None:
        logger = logging.getLogger("dataset_stats_logger")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    train_stream = benchmark.train_stream
    val_stream = benchmark.valid_stream
    test_stream = benchmark.test_stream

    logger.info("\n=== Benchmark Stats ===\n")
    for experience in train_stream:
        task_id = experience.current_experience
        classes = experience.classes_in_this_experience
        logger.info(f"Task {task_id} | Classes in this task: {classes}")

        train_dataset = experience.dataset
        logger.info(f"  Training examples: {len(train_dataset)}")

        val_dataset = val_stream[task_id].dataset
        logger.info(f"  Validation examples: {len(val_dataset)}")

        test_dataset = test_stream[task_id].dataset
        logger.info(f"  Test examples: {len(test_dataset)}")

        logger.info("-" * 50)


def factory_benchmark(name: str, seed: int) -> NCScenario:
    name = name.lower()

    benchmark_map = {
        "splitmnist": (SplitMNIST, 5, list(range(0, 10))),
        "splitcifar10": (SplitCIFAR10, 5, list(range(0, 10))),
        "splitcifar100": (SplitCIFAR100, 10, list(range(0, 100))),
        "splitcifar110": (SplitCIFAR110, 10, list(range(0, 100))),
    }

    if name not in benchmark_map:
        raise ValueError(f"Unknown dataset: {name}")

    BenchmarkClass, n_experiences, fixed_order = benchmark_map[name]
    return BenchmarkClass(n_experiences=n_experiences, seed=seed, fixed_class_order=fixed_order)
