import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import yaml

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["log"].get("output_dir", "./outputs")
os.makedirs(output_dir, exist_ok=True)

def plot_loss(train_losses, val_losses, title="Loss over epochs"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    save_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved loss plot to {save_path}")


def plot_accuracy(accuracies, title="Accuracy over epochs"):
    epochs = range(1, len(accuracies) + 1)
    plt.figure()
    plt.plot(epochs, accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    save_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved accuracy plot to {save_path}")


def plot_confusion_matrix(true_labels, pred_labels, classes=["Benign", "Malignant"], title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")
