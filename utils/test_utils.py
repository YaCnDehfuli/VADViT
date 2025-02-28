import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from config import *
from .training_utils import softmax_with_temp


def test_model(model, test_loader, device, num_classes, att_outputs=None, explainability=False):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        print("\nTest Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]))

    print("\nTest Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
