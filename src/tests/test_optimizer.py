# import numpy as np
# import yaml

# from config import Config
# from objective.manage_benchmark import get_benchmark
# from optimizers.optimizer import Optimizer


# def test_create_random_solution():
#     with open("test_config.yaml", "r") as file:
#         data = yaml.safe_load(file)

#     config = Config(**data)

#     benchmark = get_benchmark(config.dataset)

#     op = Optimizer(benchmark, config.model, config.strategies[0])
#     solution = op._create_random_solution()

#     assert isinstance(solution, dict)

#     assert len(solution) == len(4)

#     for task_id, mask in solution.items():
#         assert isinstance(mask, np.ndarray)

#         assert len(mask) == 500

#         assert len(mask) == len(set(mask))

#         assert mask.min() >= 0
#         assert mask.max() < op.task_sizes[task_id]
