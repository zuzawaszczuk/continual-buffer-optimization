# from typing import Dict, Tuple

# import numpy as np

# from .optimizer import Optimizer


# class GeneticAlgorithm(Optimizer):
#     def __init__(self, benchmark, model_config, hyperparams):
#         super().__init__(benchmark, model_config, hyperparams)
#         self.n_calls = self.hyperparams.get("n_calls", 50)
#         self.population = self.hyperparams.get("population", 50)
#         self.population = [Chromosome(chromosome_length) for _ in range(population_size)]
#         self.steps_per_restart = max(1, self.n_calls // self.restarts)

#     def optimize(self) -> Tuple[float, Dict[int, np.ndarray]]:
#         trace = []
#         for _ in range(self.num_steps):
#         self.tournament_selection()

#         for i in range(0, len(self.population) - 1, 2):
#             self.population[i], self.population[i+1] = self.reproduce((self.population[i], self.population[i+1]))

#         for individual in self.population:
#             individual.mutation(self.mutation_probability)

#         best = min(self.population, key=self.eval_objective_func)

#         print(f'Step {_}: {self.to_point(best)}')
#         trace.append(self.to_point(best))

#         self.plot_func(trace)
#         print(f'Minimum {min(trace, key=lambda x: x[2])}')

# # krzyżowanie buforów kilka masek, losowanie 250 z jednej i 250 z drugiej grupy indeksow + brakujace
# # mutacja
