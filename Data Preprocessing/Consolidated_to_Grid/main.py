import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from process2image import ProcessVisulaizer  
from config import *

def process_family(family_path, family_label, output_dir, visualizer):
    
    output_family_dir = os.path.join(output_dir, family_label) 
    os.makedirs(output_family_dir, exist_ok=True)
    # 
    try:
        hash_folders = [os.path.join(family_path, hf) for hf in os.listdir(family_path) if os.path.isdir(os.path.join(family_path, hf))]
        for hash_folder in tqdm(hash_folders, desc=f"Processing {family_label}"):
            output_file = os.path.join(output_family_dir, f"{os.path.basename(hash_folder)}.png")
            visualizer.proc_to_img(hash_folder, output_file)

    except Exception as e:
        print(f"Error processing family {family_label}: {e}")


def generate_dataset(dataset_dir, output_dir, visualizer):
    families = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    os.makedirs(output_dir, exist_ok=True)

    for family in families:
        family_path = os.path.join(dataset_dir, family)
        print(f"Processing family: {family}")
        process_family(family_path, family, output_dir, visualizer)


# Main function
if __name__ == "__main__":
    vis = ProcessVisulaizer(PATCH_SIZE, IMAGE_SIZE, TAG_MAPPING, PROTECTION_MAPPING, 'DYNAMIC')
    dataset_folder = os.path.join(IMAGE_DATASET_DIR,f"{PATCH_SIZE}_{IMAGE_SIZE}")
    os.makedirs(dataset_folder, exist_ok = True)
    generate_dataset(CONSOLIDATED_DIR, dataset_folder, vis)