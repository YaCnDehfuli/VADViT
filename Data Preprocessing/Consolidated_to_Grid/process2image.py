import os
import pandas as pd
import numpy as np
from PIL import Image
import math
import multiprocessing as mp
from tqdm import tqdm
import mmap
import gc
import re

class ProcessVisulaizer():
    def __init__(self, patch_size, image_size, tag_mapping, protection_mapping, ent_method):
        self.patch_size = patch_size
        self.image_size = image_size
        self.tag_mapping = tag_mapping
        self.protection_mapping = protection_mapping
        self.ent_method = ent_method
        self.memory_regex = re.compile(r'vad\.0x([0-9a-fA-F]+)')
        self.grid_size = int(self.image_size / self.patch_size)
        self.num_patches = self.grid_size * self.grid_size

    def extract_memory_address(self, filename):
        match = self.memory_regex.search(filename)
        if match:
            return int(match.group(1), 16)
        return None

    def downsample(self, image, method='sum'):
        h, w = image.shape
        assert h == w, "Input image must be square"
        assert h % self.patch_size == 0, "Final size must be a divisor of the original size"
        
        factor = h // self.patch_size  # Compute downsampling factor
        
        # Reshape into blocks of size (factor x factor) and aggregate
        reshaped = image.reshape(self.patch_size, factor, self.patch_size, factor)
        
        if method == 'sum':
            downsampled = reshaped.sum(axis=(1, 3))
        elif method == 'mean':
            downsampled = reshaped.mean(axis=(1, 3))
        else:
            raise ValueError("Unsupported method. Use 'sum' or 'mean'")    
        return downsampled


    def generate_feature_image(self, tag, protection):
        base_value = self.tag_mapping.get(tag, 0)
        adjustment = self.protection_mapping.get(protection, 0)
        pixel_value = base_value + adjustment
        return np.full((self.patch_size, self.patch_size), pixel_value, dtype=np.uint8)

# Entropy image generation functions
    def calculate_entropy(self, data):
        if len(data) == 0:
            return 0
        probabilities = np.bincount(data, minlength=256) / len(data)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))


    def generate_entropy_image(self, binary_data):
        num_entropy_values = self.patch_size * self.patch_size
        
        if self.ent_method == 'DYNAMIC':
            window_size = math.ceil(len(binary_data) / num_entropy_values)
        elif self.ent_method == 'STATIC':
            window_size = 128
        else:
            raise ValueError("Unsupported method. Use 'STATIC' or 'DYNAMIC'")
        
        # Ensure binary_data length is a multiple of window_size by padding with null bytes
        pad_length = (window_size - (len(binary_data) % window_size)) % window_size
       
        entropy_values = [self.calculate_entropy(binary_data[i:i + window_size]) for i in range(0, len(binary_data), window_size)]
        binary_data = np.pad(binary_data, (0, pad_length), 'constant', constant_values=0)
       
        # Ensure exactly num_entropy_values by padding or truncating
        entropy_values = (entropy_values + [0] * num_entropy_values)[:num_entropy_values]
        entropy_array = np.array(entropy_values).reshape((self.patch_size, self.patch_size))

        if self.ent_method == 'STATIC':
            entropy_array = self.downsample(entropy_array, self.patch_size, method='mean')
        
        normalized = ((entropy_array - np.min(entropy_array)) /
                    (np.max(entropy_array) - np.min(entropy_array) + 1e-9)) * 255
        return normalized.astype(np.uint8)

    # Markov image generation functions
    def compute_byte_tables(self, data):
        frequency_table = np.zeros((256, 256), dtype=np.int32)
        np.add.at(frequency_table, (data[:-1], data[1:]), 1)
        log_freq = np.log1p(frequency_table)
        max_per_row = log_freq.max(axis=1, keepdims=True)
        scaled_freq = np.divide(log_freq, max_per_row, out=np.zeros_like(log_freq), where=max_per_row != 0)
        row_sums = scaled_freq.sum(axis=1, keepdims=True)
        probability_table = np.divide(scaled_freq, row_sums, out=np.zeros_like(scaled_freq), where=row_sums != 0)
        return probability_table

    def generate_markov_image(self, binary_data):
        probability_table = self.compute_byte_tables(binary_data)
        stretched = (probability_table - probability_table.min()) / (probability_table.max() - probability_table.min())
        enhanced = np.sqrt(stretched) * 255
        hist_eq = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())  # Re-normalize
        color_map = (hist_eq * 255).astype(np.uint8)
        img_array = np.array(color_map)
        downsampled = self.downsample(img_array, method='mean')
        return downsampled.astype(np.uint8)
        # return enhanced[:32, :32].astype(np.uint8)

    # Read CSV once and create a mapping
    def read_csv_once(self,csv_path):
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            file_name = row['File output']
            tag = str(row['Tag']).strip()  # Strip whitespace
            protection = str(row['Protection']).strip()  # Strip whitespace       
            mapping[file_name] = (tag, protection)

        return mapping

    # Process a single file
    def region_to_patch(self, file_path, mapping=None):
        try:
            file_name = os.path.basename(file_path)
            addr = self.extract_memory_address(file_name)
            if file_name not in mapping:
                print(f"Warning: {file_name} not found in the mapping CSV. Skipping.")
                return

            tag, protection = mapping[file_name]

            with open(file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    binary_data = np.frombuffer(mmapped_file[:], dtype=np.uint8)
            
            feature_image = self.generate_feature_image(tag, protection)
            entropy_image = self.generate_entropy_image(binary_data)
            markov_image = self.generate_markov_image(binary_data)


            rgb_image = np.stack([feature_image, entropy_image, markov_image], axis=-1)
            return (addr, rgb_image)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def proc_to_img(self, input_dir, output_file, num_workers=8):
        patches = []
        valid_subfolders = ["malware_executable", "dlls"]

        for subfolder in valid_subfolders:
            subfolder_path = os.path.join(input_dir, subfolder)
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
            if not csv_files:
                print(f"No CSV file found in {subfolder_path}. Skipping.")
                continue

            csv_path = os.path.join(subfolder_path, csv_files[0])
            mapping = self.read_csv_once(csv_path)

            file_paths = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if not f.endswith('.csv')]
            args = [(file_path, mapping) for file_path in file_paths]

            with mp.Pool(processes=num_workers) as pool:
                patches_results = list(tqdm(pool.starmap(self.region_to_patch, args), total=len(file_paths)))

            patches_results = sorted([p for p in patches_results if p], key=lambda x: x[0])
            patches.extend([p[1] for p in patches_results])

            del mapping, file_paths, patches_results  # Free memory
            gc.collect()

        if len(patches) < self.num_patches:
            patches.extend([np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)] * (self.num_patches - len(patches)))
        elif len(patches) > self.num_patches:
            patches = patches[:self.num_patches]

        grid_image = Image.new("RGB", (self.grid_size * self.patch_size, self.grid_size * self.patch_size))
        for idx, patch_array in enumerate(patches):
            patch = Image.fromarray(patch_array.astype(np.uint8))
            row, col = divmod(idx, self.grid_size)
            grid_image.paste(patch, (col * self.patch_size, row * self.patch_size))

        grid_image.save(output_file)
        return grid_image