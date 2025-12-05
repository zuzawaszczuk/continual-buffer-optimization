from abc import ABC, abstractmethod
import numpy as np
from buffer import Buffer
import torch

class SelectionStrategy(ABC):
    @abstractmethod
    def update_buffer(self, buffer: Buffer, data, labels, task_id):
        pass

class RandomStrategy(SelectionStrategy):
    def update_buffer(self, buffer: Buffer, data, labels, task_id):
        remaining_data, remaining_labels, remaining_task_id = buffer.add_data(data, labels, task_id)

        if  remaining_data is not None and remaining_labels is not None and remaining_task_id is not None:
            n_candidates = remaining_data.shape[0]
            
            safe_ceiling = int(buffer.max_size * 0.5) 
            if safe_ceiling < 1: safe_ceiling = 1 
            
            n_random_target = np.random.randint(1, safe_ceiling + 1)
            n_to_insert = min(n_candidates, n_random_target)
            
            candidate_indices = torch.randperm(n_candidates)[:n_to_insert]
            
            data_subset = remaining_data[candidate_indices]
            labels_subset = remaining_labels[candidate_indices]
            task_ids_subset = remaining_task_id[candidate_indices]
            indices_to_overwrite = np.random.choice(buffer.max_size, n_to_insert, replace=False)
            
            buffer.replace_at_indices(indices_to_overwrite, data_subset, labels_subset, task_ids_subset)
