# utils/training_utils.py
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from config import *

def softmax_with_temp(logits, temp=0.7):
    return F.softmax(logits / temp, dim=-1)


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, visualizer, device, num_epochs=40, num_classes=4, save_path="./224_best_model.pt"):
    swa_model = None
    swa_scheduler = None
    best_val_accuracy = 0.0  # Track the best validation accuracy
    unfreeze_epochs = [6 * (i + 1) for i in range(STEPS)]
    layers_per_step = FROZEN_LAYERS // STEPS

    for epoch in range(num_epochs):
        
        if epoch in unfreeze_epochs:                 # Dynamic Layer Freezing
            step_idx = unfreeze_epochs.index(epoch)  # Get current step index
            start_layer = FROZEN_LAYERS - (step_idx + 1) * layers_per_step
            end_layer = start_layer + layers_per_step

            print(f"Unfreezing layers {start_layer} to {end_layer}")
            for param in model.vit.blocks[start_layer:end_layer].parameters():
                param.requires_grad = True  

        # Training Phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            outputs = softmax_with_temp(outputs, temp=0.7)
            _, preds = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()

        # Calculate Training Accuracy
        train_accuracy = train_correct / train_total
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)              
                
                loss = criterion(outputs, labels)  # Compute validation loss
                total_val_loss += loss.item()

                outputs = softmax_with_temp(outputs, temp=0.7)  # Adjust temp as needed
                _, preds = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calculate Validation Accuracy
        val_accuracy = val_correct / val_total
        avg_val_loss = total_val_loss / len(val_loader)
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        visualizer.update(epoch, train_accuracy, val_accuracy, avg_train_loss, avg_val_loss, macro_precision, macro_recall, macro_f1)

        print(f"Validation Accuracy for Epoch {epoch+1}: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"\nValidation Classification Report for Epoch {epoch+1}:")
        print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]))
        print(f"\nValidation Confusion Matrix for Epoch {epoch+1}:")
        print(confusion_matrix(all_labels, all_preds))   

        if epoch >= 32:
            if swa_model is None:
                current_lr = optimizer.param_groups[0]['lr']  # Get last adjusted LR from ReduceLROnPlateau
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=current_lr * 0.5, anneal_strategy="cos", anneal_epochs=5)
            swa_model.update_parameters(model)
            swa_scheduler.step()
            print(f"🔄 SWA Model Updated at Epoch {epoch}")

        # Update the learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Save the best model
        if (val_accuracy > best_val_accuracy): # and (val_accuracy > 0.96):
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    visualizer.save_plot()
    visualizer.close()
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}. Model saved to {save_path}")
