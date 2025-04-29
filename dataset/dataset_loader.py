# dataset/dataset_loader.py

import os
import random
from torch.utils.data import Dataset
from PIL import Image
from .transforms import train_transform, val_transform


def build_label_mapping(dataset_dir, multiclass):
    """Map family folder names to class labels. Deterministic for a given directory."""
    if multiclass:
        return {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(dataset_dir)))}

    family_to_label = {"Benign": 0}
    for class_name in sorted(os.listdir(dataset_dir)):
        if class_name.lower() != "benign":
            family_to_label[class_name] = 1
    return family_to_label


def discover_samples(dataset_dir, family_to_label):
    """Enumerate every (image_path, label) pair in a fixed, filesystem-independent order."""
    all_samples = []
    for class_name, label in family_to_label.items():
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            for image_name in sorted(os.listdir(class_path)):
                if image_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)
                    all_samples.append((image_path, label))

    # Sorting by path makes the ordering independent of os.listdir() ordering,
    # so the partition below is reproducible across machines and runs.
    all_samples.sort(key=lambda sample: sample[0])
    return all_samples


def partition_samples(all_samples, train_ratio, val_ratio, test_ratio, random_seed):
    """Split the samples once into three mutually exclusive subsets.

    The shuffle is driven by a private random.Random instance seeded with
    random_seed, so the partition depends only on (samples, ratios, seed) and
    never on the global RNG state. Every ImageDataset built with the same
    arguments therefore sees the same partition, and the three splits are
    complementary slices of a single permutation.
    """
    order = list(range(len(all_samples)))
    random.Random(random_seed).shuffle(order)

    total_size = len(order)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    split_indices = {
        "train": order[:train_size],
        "val": order[train_size:train_size + val_size],
        "test": order[train_size + val_size:],
    }
    return {name: [all_samples[i] for i in idx] for name, idx in split_indices.items()}


class ImageDataset(Dataset):
    def __init__(self, dataset_dir, num_classes, multiclass = False, split="train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"

        self.dataset_dir = dataset_dir
        self.num_classes = num_classes
        self.split = split
        self.samples = []

        # Map folder names to class labels dynamically
        self.family_to_label = build_label_mapping(dataset_dir, multiclass)
        print(f" Label Mapping: {self.family_to_label}")

        all_samples = discover_samples(dataset_dir, self.family_to_label)
        print(f"Total images found: {len(all_samples)} for split {split}")

        splits = partition_samples(all_samples, train_ratio, val_ratio, test_ratio, random_seed)

        # Guard against a regression ever reintroducing overlapping splits.
        train_paths = {path for path, _ in splits["train"]}
        val_paths = {path for path, _ in splits["val"]}
        test_paths = {path for path, _ in splits["test"]}
        assert not (train_paths & val_paths), "train/val splits overlap"
        assert not (train_paths & test_paths), "train/test splits overlap"
        assert not (val_paths & test_paths), "val/test splits overlap"

        self.samples = splits[split]
        self.transform = train_transform if split == "train" else val_transform

        class_counts = {i: 0 for i in range(num_classes)}
        for _, label in self.samples:
            class_counts[label] += 1
        print(f"📊 {split} dataset class distribution: {class_counts}")

    def image_paths(self):
        """Stable identities for the samples in this split, for leakage checks."""
        return [path for path, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label
