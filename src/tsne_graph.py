import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import yaml
import torch
import os
from config import Config
from objective.manage_benchmark import get_benchmark

INPUT_DIR = "SplitMNIST_100_ncall_2026-01-21 18:55"
MASKS_FILE = INPUT_DIR + "/masks.pkl"
CONFIG_FILE = INPUT_DIR + "/history_config.yaml"
SAMPLES_FOR_TSNE = 30000

def load_config_and_benchmark():
    with open(CONFIG_FILE, "r") as file:
        data_config = yaml.safe_load(file)
    config = Config(**data_config)
    benchmark = get_benchmark(config.dataset)
    return benchmark

def get_all_data(benchmark, task_ids):
    X_list = []
    y_list = []
    task_offsets = {}
    current_offset = 0
    
    for t_id in task_ids:
        try:
            dataset = benchmark.train_stream[t_id].dataset
        except IndexError:
            continue  
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        batch = next(iter(loader))
        X_curr = batch[0].view(batch[0].size(0), -1).numpy()
        y_curr = batch[1].numpy()

        X_list.append(X_curr)
        y_list.append(y_curr)
        task_offsets[t_id] = current_offset
        current_offset += len(X_curr)
                
    X_global = np.concatenate(X_list, axis=0)
    y_global = np.concatenate(y_list, axis=0)
    
    return X_global, y_global, task_offsets

def compute_tsne_smart(X, mandatory_indices, total_samples=30000):
    total_len = len(X)
    mandatory_indices = np.array(list(mandatory_indices))
    remaining_slots = total_samples - len(mandatory_indices)
    
    if remaining_slots < 0:
        print(f"Buffer size greater than rem slots")
        indices = mandatory_indices
    else:
        all_indices = np.arange(total_len)
        background_candidates = np.setdiff1d(all_indices, mandatory_indices)
        if len(background_candidates) > remaining_slots:
            background_indices = np.random.choice(background_candidates, remaining_slots, replace=False)
        else:
            background_indices = background_candidates
            
        indices = np.concatenate([mandatory_indices, background_indices])
        indices.sort()
        
    X_subset = X[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=35, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_subset)
    
    return X_embedded, indices

def plot_global_strategies(X_embedded, background_indices, masks_dict, y_global, task_offsets):
    strategies = list(masks_dict.keys())
    
    cols = 2
    rows = (len(strategies) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    axes = axes.flatten()
    
    real_to_embedded = {real_idx: emb_idx for emb_idx, real_idx in enumerate(background_indices)}
    palette = sns.color_palette("tab10", 10)

    for i, strategy_name in enumerate(strategies):
        ax = axes[i]
        
        ax.scatter(
            X_embedded[:, 0], X_embedded[:, 1],
            c='lightgray', s=5, alpha=0.2, label='All Data'
        )
        
        strat_masks = masks_dict[strategy_name]
        total_buffer_points = 0
        plotted_points = 0
        
        for t_id, buffer_indices in strat_masks.items():
            if t_id not in task_offsets:
                continue
            offset = task_offsets[t_id]
            global_indices = [idx + offset for idx in buffer_indices]

            embedding_indices = []
            valid_global_indices = []
            for g_idx in global_indices:
                if g_idx in real_to_embedded:
                    embedding_indices.append(real_to_embedded[g_idx])
                    valid_global_indices.append(g_idx)
            
            if not embedding_indices:
                continue
                
            buffer_x = X_embedded[embedding_indices, 0]
            buffer_y = X_embedded[embedding_indices, 1]
            buffer_labels = y_global[valid_global_indices]
            
            unique_labels_in_buffer = np.unique(buffer_labels)
            for cls in unique_labels_in_buffer:
                mask_cls = buffer_labels == cls
                ax.scatter(
                    buffer_x[mask_cls], buffer_y[mask_cls],
                    color=palette[int(cls)],
                    s=25, edgecolor='black', linewidth=0.3, alpha=0.9
                )
            
            total_buffer_points += len(buffer_indices)
            plotted_points += len(embedding_indices)

        ax.set_title(f"{strategy_name}", fontsize=14)
        ax.axis('off')

    for j in range(len(strategies), len(axes)):
        axes[j].axis('off')

    active_classes = np.unique(y_global)
    active_classes.sort()
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[int(cls)], 
                   markersize=10, label=f'Digit {int(cls)}') 
        for cls in active_classes
    ]
    
    fig.legend(handles=handles, loc='center right', title="Classes")
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    filename = "tsne_graph.png"
    plt.savefig(filename, dpi=512)
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(MASKS_FILE):
        print("No file found.")
        exit()

    with open(MASKS_FILE, 'rb') as f:
        masks = pickle.load(f)
    
    benchmark = load_config_and_benchmark()
    first_strat = list(masks.keys())[0]
    
    task_ids = sorted(list(masks[first_strat].keys()))    
    X_global, y_global, offsets = get_all_data(benchmark, task_ids)
    
    all_buffer_global_indices = set()
    
    for strategy_name, strat_masks in masks.items():
        for t_id, buffer_indices in strat_masks.items():
            if t_id in offsets:
                offset = offsets[t_id]
                for idx in buffer_indices:
                    all_buffer_global_indices.add(idx + offset)
    X_emb, bg_indices = compute_tsne_smart(X_global, all_buffer_global_indices, total_samples=SAMPLES_FOR_TSNE)
    
    plot_global_strategies(X_emb, bg_indices, masks, y_global, offsets)