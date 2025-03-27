import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self, num_epochs, save_path="training_plot.png"):
        self.num_epochs = num_epochs
        self.save_path = save_path
        
        self.epochs = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []
        
        # New lists for macro precision, recall, and F1-score
        self.macro_precisions = []
        self.macro_recalls = []
        self.macro_f1_scores = []

        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 15))  # Three separate subplots

    def update(self, epoch, train_acc, val_acc, train_loss, val_loss, precision, recall, f1):
        self.epochs.append(epoch + 1)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.macro_precisions.append(precision)
        self.macro_recalls.append(recall)
        self.macro_f1_scores.append(f1)

        # --- First Plot: Training & Validation Accuracy ---
        self.ax1.clear()
        self.ax1.plot(self.epochs, self.train_accuracies, label='Train Accuracy', color='#2ca02c', marker='o', linestyle='-')  # 🟢 Green
        self.ax1.plot(self.epochs, self.val_accuracies, label='Val Accuracy', color='#1f77b4', marker='o', linestyle='-')  # 🔵 Blue

        self.ax1.set_ylabel("Accuracy")
        self.ax1.set_ylim(0, 1)  # Fixed range for accuracy
        self.ax1.legend(loc="lower right")
        self.ax1.set_title("Training & Validation Accuracy")
        self.ax1.grid(True)

        # --- Second Plot: Training & Validation Loss ---
        self.ax2.clear()
        self.ax2.plot(self.epochs, self.train_losses, label='Train Loss', color='#000000', marker='s', linestyle='--', alpha=0.7)  # ⚫ Black (dashed)
        self.ax2.plot(self.epochs, self.val_losses, label='Val Loss', color='#d62728', marker='s', linestyle='--', alpha=0.7)  # 🔴 Red (dashed)

        self.ax2.set_ylabel("Loss")
        self.ax2.set_ylim(0, 2.2)  # Fixed range for loss
        self.ax2.legend(loc="upper right")
        self.ax2.set_title("Training & Validation Loss")
        self.ax2.grid(True)

        # --- Third Plot: Validation Metrics (Macro Precision, Recall, F1-Score, Accuracy) ---
        self.ax3.clear()
        self.ax3.plot(self.epochs, self.val_accuracies, label="Val Accuracy", color='#1f77b4', marker='o', linestyle='-')  # 🔵 Blue
        self.ax3.plot(self.epochs, self.macro_precisions, label="Macro Precision", color='#ff7f0e', marker='s', linestyle='--')  # 🟡 Golden Yellow
        self.ax3.plot(self.epochs, self.macro_recalls, label="Macro Recall", color='#d62728', marker='d', linestyle='-.')  # 🔴 Crimson Red (dashed)
        self.ax3.plot(self.epochs, self.macro_f1_scores, label="Macro F1-Score", color='#9467bd', marker='^', linestyle='-')  # 🟣 Violet

        self.ax3.set_xlabel("Epoch")
        self.ax3.set_ylabel("Score")
        self.ax3.set_ylim(0, 1)  # Fixed range
        self.ax3.legend()
        self.ax3.set_title("Validation Accuracy, Macro Precision, Recall, & F1-Score")
        self.ax3.grid(True)

        self.fig.tight_layout()  # Ensure proper spacing between plots
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_plot(self):
        self.fig.savefig(self.save_path)
        print(f"Training plot saved to {self.save_path}")

    def close(self):
        plt.ioff()
        plt.show()
