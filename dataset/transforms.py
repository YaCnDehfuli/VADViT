# dataset/transforms.py

import torchvision.transforms as T
from config import IMAGE_SIZE

# Training transformations
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.4),
    T.ColorJitter(contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation & Test transformations (no augmentation)
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
