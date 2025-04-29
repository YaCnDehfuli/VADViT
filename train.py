# # train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from config import *
from models.ViT_model import ViTForImages
from dataset.dataset_loader import ImageDataset
from utils.training_utils import train_and_validate
from utils.seed import set_seed
from utils.metrics_visulaization import TrainingVisualizer


def main():
    set_seed(SEED)

    # Load dataset
    train_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="train")
    val_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="val")
    test_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="test")

    # Check for data leakage on stable sample identities (image paths). Comparing
    # pixel statistics is unreliable: augmentation changes them for a single image
    # and distinct images can collide. Abort rather than warn, so a leaking split
    # can never silently produce reported metrics.
    train_paths = set(train_dataset.image_paths())
    val_paths = set(val_dataset.image_paths())
    test_paths = set(test_dataset.image_paths())

    leaks = {
        "train/val": train_paths & val_paths,
        "train/test": train_paths & test_paths,
        "val/test": val_paths & test_paths,
    }
    leaking = {name: shared for name, shared in leaks.items() if shared}
    if leaking:
        details = "; ".join(f"{name}: {len(shared)} shared samples" for name, shared in leaking.items())
        raise RuntimeError(f"Data leakage detected between splits ({details}). Aborting training.")

    print(f"No data leakage: {len(train_paths)} train / {len(val_paths)} val / "
          f"{len(test_paths)} test samples are mutually disjoint.")


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTForImages(MODEL_NAME, num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters())   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.33, verbose=True)
    visualizer = TrainingVisualizer(NUM_EPOCHS)

    # Train the model
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, visualizer, device, NUM_EPOCHS, NUM_CLASSES, SAVE_PATH)

if __name__ == "__main__":
    main()
