# dataset/transforms.py

import torchvision.transforms as T
import torch
import random
from config import IMAGE_SIZE

# Custom augmentation: Gaussian noise
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            img_tensor = T.ToTensor()(img)
            noise = torch.randn_like(img_tensor) * self.std + self.mean
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
            return T.ToPILImage()(img_tensor)
        return img

# Training transformations (augmentations included)
train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomApply([T.RandomPosterize(bits=4)], p=0.4),
    T.RandomApply([T.RandomSolarize(threshold=128)], p=0.4),
    T.RandomApply([T.RandomInvert()], p=0.3),
    T.RandomApply([T.RandomEqualize()], p=0.4),
    T.RandomApply([T.RandomAutocontrast()], p=0.4),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.4),
    T.ColorJitter(contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.65),
    AddGaussianNoise(mean=0.0, std=0.02, p=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation & Test transformations (no augmentation)
val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
