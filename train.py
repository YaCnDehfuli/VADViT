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
import numpy as np


def main():
    set_seed(SEED)

    # Load dataset
    train_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="train")
    val_dataset = ImageDataset(DATASET_PATH, NUM_CLASSES, MULTICLASSS, split="val")

    # Convert datasets into a format that allows hashing
    train_data, _ = zip(*[train_dataset[i] for i in range(len(train_dataset))])  # Extract images
    val_data, _ = zip(*[val_dataset[i] for i in range(len(val_dataset))])  # Extract images

    # Convert to NumPy arrays for quick comparison
    train_hash = np.array([np.sum(img.numpy()) for img in train_data])
    val_hash = np.array([np.sum(img.numpy()) for img in val_data])

    # Check for data leakage
    overlap = np.intersect1d(train_hash, val_hash)

    if len(overlap) > 0:
        print("Data leakage detected! Training and validation sets are not fully separate.")
    else:
        print("No data leakage found!")


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
