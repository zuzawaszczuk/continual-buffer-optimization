import torch
import numpy as np

class Buffer:
    """
    Class implementing a simple fixed-size buffer for continual learning.

    :attr examples: Tensor holding the input examples.
    :attr labels: Tensor holding the corresponding labels.
    :attr task_ids: Tensor holding the task identifiers (if applicable).

    :attr max_size: Maximum number of samples the buffer can hold.
    :attr current_size: Current number of samples stored in the buffer.
    :attr device: Device where the buffer tensors are stored. (cpu / cuda)
    """
    def __init__(self, max_size, input_shape, device):
        """
        COnstructor foor the Buffer class.
        
        :param max_size: Maximum size of the buffer.
        :param input_shape: Shape of the input data.
        :param num_classes: Number of classes in the dataset.
        :param device: Device where the buffer tensors will be stored.
        """
        self.max_size = max_size
        self.device = device
        self.current_size = 0
        
        # create empty buffer
        self.examples = torch.zeros((max_size, *input_shape), device=device)
        self.labels = torch.zeros((max_size), dtype=torch.long, device=device)
        self.task_ids = torch.zeros((max_size), dtype=torch.long, device=device)

    def add_data(self, data, labels, task_id=None):
        """
        Adds new data to the buffer until it is full. 
        If there are samples that dont fit, they are returned.
        """
        batch_size = data.shape[0]  # number of samples to add
        if self.current_size < self.max_size:
            to_add = min(batch_size, self.max_size - self.current_size) # to_add - max number of samples we can possibly add
            
            start = self.current_size
            end = self.current_size + to_add
            
            self.examples[start:end] = data[:to_add].to(self.device)
            self.labels[start:end] = labels[:to_add].to(self.device)
            if task_id is not None:
                self.task_ids[start:end] = task_id[:to_add].to(self.device)
            
            self.current_size += to_add

            if batch_size > to_add:
                return data[to_add:], labels[to_add:], task_id[to_add:] if task_id is not None else None # if there's leftover data, return it
            
            return None, None, None
            
        return data, labels, task_id # buffer is full, return all data

    def replace_at_indices(self, i, data, labels, task_id=None):
        """
        Replaces data at specified indices in the buffer.
        :param i: Indices where data should be replaced. Can be a list or array of indices.
        """
        data, labels = data.to(self.device), labels.to(self.device)
        self.examples[i] = data
        self.labels[i] = labels
        if task_id is not None:
            self.task_ids[i] = task_id.to(self.device)

    def get_data(self):
        """
        Return all data currently stored in the buffer as a tuple: (examples, labels, task_ids).
        """
        if self.current_size == 0:
            return None, None, None
        
        return (self.examples[:self.current_size].clone(), 
                self.labels[:self.current_size].clone(), 
                self.task_ids[:self.current_size].clone())
    
    def is_empty(self):
        return self.current_size == 0
    
    def clear(self):
        """
        Clears the buffer.
        """
        self.current_size = 0
    
    def print_data(self):
        """
        Prints the contents of the buffer.
        """
        print("Buffer contents:")
        for i in range(self.current_size):
            print(f"Index {i}: Label={self.labels[i].item()}, Task ID={self.task_ids[i].item()}")

        unique, counts = torch.unique(self.labels[:self.current_size], return_counts=True)
        print("Class distribution in buffer:")
        for u, c in zip(unique, counts):
            print(f"Class {u.item()}: {c.item()} samples")
        

    def __len__(self):
        return self.current_size 