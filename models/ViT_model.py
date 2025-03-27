import torch
import torch.nn as nn
from timm import create_model
from config import *

class ViTForImages(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        self.vit = create_model(base_model_name, pretrained=True)
        self.vit.head = nn.Linear(self.vit.num_features, num_classes)

        # Freeze first 6 transformer blocks
        for param in self.vit.blocks[:FROZEN_LAYERS].parameters():
            param.requires_grad = False  

    def forward(self, x):
        return self.vit(x)
