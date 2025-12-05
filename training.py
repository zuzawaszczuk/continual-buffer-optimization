import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
from typing import cast
from torch.utils.data import Dataset

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleMLP
from buffer import Buffer
from strategies import RandomStrategy

def evaluate_model(model, test_stream, current_task_idx, device):
    """
    Testuje model na sumie wszystkich dotychczasowych zadań (Cumulative Accuracy).
    Dodatkowo wypisuje szczegóły dla każdego zadania osobno.
    """
    model.eval()
    
    datasets_so_far = []
    classes_so_far = []
    
    print(f"\n--- EVALUATE, TASK {current_task_idx} ---")
    
    # separate testing for each task
    for i in range(current_task_idx + 1):
        experience = test_stream[i]
        datasets_so_far.append(experience.dataset)
        classes_so_far.extend(experience.classes_in_this_experience)
        
        loader = DataLoader(experience.dataset, batch_size=256, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels, _ in loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Task {i} (classes: {experience.classes_in_this_experience}): {100 * correct / total:.2f}%")

    # cumulative testing (class incremental)
    if current_task_idx != 0:
        casted_datasets = [cast(Dataset, d) for d in datasets_so_far]
        cumulative_dataset = ConcatDataset(casted_datasets)
        cumulative_loader = DataLoader(cumulative_dataset, batch_size=256, shuffle=False)
        cumulative_correct = 0
        cumulative_total = 0
        
        with torch.no_grad():
            for data, labels, _ in cumulative_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                cumulative_total += labels.size(0)
                cumulative_correct += (predicted == labels).sum().item()
                
        avg_acc = 100 * cumulative_correct / cumulative_total
        print(f"Overall accuracy (classes {sorted(list(set(classes_so_far)))}): {avg_acc:.2f}%")
        print("--------------------------------------------------")


def main():
    # configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")

    USE_BUFFER = True # to-do: implement it in a better way
    BUFFER_MAX_SIZE = 500

    model = SimpleMLP(num_classes=10).to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()

    scenario = SplitFMNIST(
        n_experiences=5, 
        seed=67, 
        fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
    )
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream
    
    buffer = Buffer(max_size=BUFFER_MAX_SIZE, input_shape=(1, 28, 28), device=device)
    strategy = RandomStrategy()

    # starting experiment
    for task_idx, experience in enumerate(train_stream[:3]):
        # prepare data
        task_id = experience.current_experience
        print(f"\n- Task {task_id} (classes: {experience.classes_in_this_experience}) -")
        
        dataset_casted = cast(Dataset, experience.dataset)
        temp_loader = DataLoader(dataset_casted, batch_size=len(experience.dataset), shuffle=False)
        new_data, new_labels, new_task_ids = next(iter(temp_loader))
        new_data, new_labels, new_task_ids = new_data.to(device), new_labels.to(device), new_task_ids.to(device)
        current_dataset = TensorDataset(new_data, new_labels, new_task_ids)
        
        buf_data, buf_labels, buf_task_ids = None, None, None
        
        if USE_BUFFER and not buffer.is_empty():
            buf_data, buf_labels, buf_task_ids = buffer.get_data()
        
        if buf_data is not None and buf_labels is not None and buf_task_ids is not None:
            buffer_dataset = TensorDataset(buf_data, buf_labels, buf_task_ids)
            combined_dataset = ConcatDataset([current_dataset, buffer_dataset])
        else:
            combined_dataset = current_dataset

        train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True, num_workers=0)

        # train
        model.train()
        epochs = 2
        for epoch in range(epochs):
            # animated progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            for data, labels, _ in progress_bar:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if USE_BUFFER:
            model.eval()
            candidate_loader = DataLoader(current_dataset, batch_size=len(experience.dataset), shuffle=True)
            cand_x, cand_y, cand_t = next(iter(candidate_loader))
            cand_t = torch.full((cand_x.shape[0],), task_id, dtype=torch.long, device=device)
            strategy.update_buffer(buffer, cand_x, cand_y, cand_t)
            # buffer.print_data()

        evaluate_model(model, test_stream, task_idx, device)

    print("\n--- END ---")

if __name__ == "__main__":
    main()