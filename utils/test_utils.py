import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from .att_visualization import overlay_attention
from config import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from .training_utils import softmax_with_temp
auc_folder = AUC_FOLDER 
cm_folder = CM_FOLDER

def plot_curve(fpr, tpr, roc_auc, thresholds, model_name):
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    for i in range(0, len(thresholds), max(1, len(thresholds) // 30)):  # Select threshold points evenly
        plt.annotate(f"{thresholds[i]:.2f}", (fpr[i], tpr[i]), 
                     textcoords="offset points", xytext=(10,-5), ha='center', fontsize=10, color='red')

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Best Threshold = {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve with Thresholds')
    plt.legend(loc="lower right")
    fig_filename = f"{model_name.split('.pt')[0]}.pdf"
    fig_path =os.path.join(auc_folder, fig_filename)  
    plt.savefig(fig_path, bbox_inches='tight')
    # plt.show()


def plot_cm(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    fig_filename = f"{model_name.split('.pt')[0]}.pdf"
    fig_path =os.path.join(cm_folder, fig_filename)  
    plt.savefig(fig_path, bbox_inches='tight')
    # plt.show()

def test_model(model, test_loader, device, num_classes, att_outputs=None, explainability=False):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = softmax_with_temp(outputs)[:, 1]  # Get probability for class 1
            _, preds = torch.max(outputs, 1)
            
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Explainability
            if explainability and att_outputs and 'attn' in att_outputs:
                cls_attention = att_outputs['attn'][:, 0, 1:].mean(dim=0)
                overlay_attention(images[0], cls_attention)

        print("\nTest Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]))
    
    print("\nTest Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plot_curve(fpr, tpr, roc_auc, thresholds, EXPERIMENT_NAME)
    plot_cm(cm, EXPERIMENT_NAME)


    
