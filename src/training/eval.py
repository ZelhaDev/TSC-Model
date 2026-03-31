"""
Traffic Sign Classifier — Evaluation
======================================
Evaluates the trained CNN on the test set.
Produces: accuracy, macro-F1, classification report,
confusion matrix heatmap, and learning curves.

Run:
    python src/eval.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from src.data.data_pipeline import load_config, get_dataloaders
from src.models.cnn import TrafficSignCNN


# GTSRB class names (abbreviated)
GTSRB_NAMES = [
    "20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h",
    "80 km/h", "End 80", "100 km/h", "120 km/h", "No passing",
    "No pass >3.5t", "Priority", "Priority road", "Yield", "Stop",
    "No vehicles", "No >3.5t", "No entry", "General caution",
    "Left curve", "Right curve", "Double curve", "Bumpy road",
    "Slippery", "Narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children", "Bicycles", "Ice/snow",
    "Wild animals", "End limits", "Right turn", "Left turn",
    "Ahead only", "Ahead/right", "Ahead/left", "Keep right",
    "Keep left", "Roundabout", "End no passing", "End no pass >3.5t",
]


@torch.no_grad()
def get_predictions(model, loader, device):
    """Return all predictions and ground-truth labels."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", ax=ax, fmt="d",
                xticklabels=range(43), yticklabels=range(43))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CNN Confusion Matrix — GTSRB Test Set")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def plot_learning_curves(history, save_path):
    """Plot train/val loss and accuracy from training history."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Val F1
    axes[2].plot(epochs, history["val_f1"], "D-", color="green",
                 label="Val Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_title("Validation Macro-F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("CNN Training Curves — GTSRB", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Learning curves saved → {save_path}")


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data ...")
    _, _, test_loader, num_classes = get_dataloaders(cfg)

    # Load best model
    ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    model = TrafficSignCNN(num_classes=num_classes,
                           dropout=cfg["cnn"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    print(f"Loaded model from {ckpt}")

    # Predictions
    preds, labels = get_predictions(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"\n{'='*50}")
    print(f"  TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=GTSRB_NAMES,
                                zero_division=0))

    # Save test metrics
    results_dir = cfg["paths"]["results"]
    logs_dir = cfg["paths"]["logs"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    test_metrics = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
    with open(os.path.join(logs_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Plots
    plot_confusion_matrix(labels, preds,
                          os.path.join(results_dir, "confusion_matrix.png"))

    # Learning curves (from training log)
    history_path = os.path.join(logs_dir, "training_metrics.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_learning_curves(history,
                             os.path.join(results_dir, "learning_curves.png"))
    else:
        print("  (No training_metrics.json found — skipping learning curves)")

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()