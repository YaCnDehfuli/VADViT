# utils/training_utils.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from config import *

def softmax_with_temp(logits, temp=0.7):
    return F.softmax(logits / temp, dim=-1)


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, visualizer, device, num_epochs=40, num_classes=4, save_path="./224_best_model.pt"):
    best_val_accuracy = 0.0  # Track the best validation accuracy

    for epoch in range(num_epochs):
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

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                outputs = softmax_with_temp(outputs, temp=0.7)
                _, preds = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_accuracy = val_correct / val_total
        avg_val_loss = total_val_loss / len(val_loader)
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        visualizer.update(epoch, train_accuracy, val_accuracy, avg_train_loss, avg_val_loss, macro_precision, macro_recall, macro_f1)

        print(f"Validation Accuracy for Epoch {epoch+1}: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(num_classes)]))
        print(confusion_matrix(all_labels, all_preds))

        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    visualizer.save_plot()
    visualizer.close()
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}. Model saved to {save_path}")
