import pickle
import torch
import torchvision
from torchvision.utils import make_grid, save_image
import yaml
import os
import random
from collections import defaultdict
from config import Config
from objective.manage_benchmark import get_benchmark

INPUT_DIR = "SplitMNIST_500_ncall_3_tasks"
MASKS_FILE = INPUT_DIR + "/masks.pkl"
CONFIG_FILE = INPUT_DIR + "/history_config.yaml"
OUTPUT_DIR = INPUT_DIR + "/buffer_preview"
SAMPLES_PER_CLASS = 100
GRID_ROW_SIZE = 10

def load_data():
    if not os.path.exists(MASKS_FILE):
        print(f"No file found")
        exit()

    with open(MASKS_FILE, 'rb') as f:
        masks = pickle.load(f)

    with open(CONFIG_FILE, "r") as file:
        data_config = yaml.safe_load(file)
    config = Config(**data_config)
    benchmark = get_benchmark(config.dataset)
    
    return masks, benchmark

def generate_grids(masks, benchmark):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    strategies = list(masks.keys())
    
    for strategy_name in strategies:
        print(f"{strategy_name}")        
        images_by_class = defaultdict(list)
        for task_id, indices in masks[strategy_name].items():
            try:
                dataset = benchmark.train_stream[task_id].dataset
            except IndexError:
                continue
            
            for idx in indices:
                img, label, _ = dataset[idx]                
                label_item = label.item() if isinstance(label, torch.Tensor) else label
                images_by_class[label_item].append(img)

        for cls, img_list in sorted(images_by_class.items()):
            count = len(img_list)
            
            if count > SAMPLES_PER_CLASS:
                selected_imgs = random.sample(img_list, SAMPLES_PER_CLASS)
            else:
                selected_imgs = img_list
            
            batch_tensor = torch.stack(selected_imgs)
            grid_img = make_grid(batch_tensor, nrow=GRID_ROW_SIZE, padding=2, normalize=True)
            filename = f"{OUTPUT_DIR}/{strategy_name}_Digit_{cls}.png"
            save_image(grid_img, filename)
            

if __name__ == "__main__":
    masks, benchmark = load_data()
    generate_grids(masks, benchmark)
