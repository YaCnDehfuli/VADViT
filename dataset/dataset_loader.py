# dataset/dataset_loader.py

import os
import random
from torch.utils.data import Dataset, random_split
from PIL import Image
from .transforms import train_transform, val_transform


class ImageDataset(Dataset):
    def __init__(self, dataset_dir, num_classes, multiclass = False, split="train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"

        self.dataset_dir = dataset_dir
        self.num_classes = num_classes
        self.split = split
        self.samples = []

        # Map folder names to class labels dynamically
        if multiclass:
            self.family_to_label = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(dataset_dir)))}
        else:
            self.family_to_label = {"Benign": 0}
            for class_name in sorted(os.listdir(dataset_dir)):
                if class_name.lower() != "benign":
                    self.family_to_label[class_name] = 1
            
        print(f" Label Mapping: {self.family_to_label}")

        all_samples = []
        for class_name, label in self.family_to_label.items():
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                for image_name in sorted(os.listdir(class_path)):
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):  
                        image_path = os.path.join(class_path, image_name)
                        all_samples.append((image_path, label))

        print(f"Total images found: {len(all_samples)} for split {split}")

        random.seed(random_seed)
        random.shuffle(all_samples)

        total_size = len(all_samples)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size  

        train_samples, val_samples, test_samples = random_split(all_samples, [train_size, val_size, test_size])

        if split == "train":
            self.samples = train_samples
            self.transform = train_transform
        elif split == "val":
            self.samples = val_samples
            self.transform = val_transform
        elif split == "test":
            self.samples = test_samples
            self.transform = val_transform

        class_counts = {i: 0 for i in range(num_classes)}
        for _, label in self.samples:
            class_counts[label] += 1
        print(f"📊 {split} dataset class distribution: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label
