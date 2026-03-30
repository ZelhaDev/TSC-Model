"""
Traffic Sign Classifier — Evaluation (Enhanced)
=================================================
Evaluates the trained CNN on the test set.
Produces: accuracy, macro-F1, classification report,
confusion matrix heatmap, learning curves, PR/ROC curves,
and a comprehensive results summary table.

Run:
    python src/eval.py
"""

import os
import json
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

from data_pipeline import load_config, get_dataloaders
from models.cnn import TrafficSignCNN


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
    """Return predictions, labels, and raw probabilities."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


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
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss",
                 markersize=3)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss",
                 markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc",
                 markersize=3)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc",
                 markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Val F1
    axes[2].plot(epochs, history["val_f1"], "D-", color="green",
                 label="Val Macro-F1", markersize=3)
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


def plot_roc_pr_curves(y_true, y_probs, save_dir, num_classes=43):
    """Plot macro-averaged ROC and PR curves, plus per-class for top/bottom-5."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # --- Macro-averaged ROC ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-class F1 to find top/bottom 5
    per_class_f1 = f1_score(y_true, y_probs.argmax(1), average=None,
                            zero_division=0)
    top5 = np.argsort(per_class_f1)[-5:]
    bottom5 = np.argsort(per_class_f1)[:5]
    highlight_classes = list(bottom5) + list(top5)

    colors = plt.cm.tab10(np.linspace(0, 1, len(highlight_classes)))

    # ROC curves for selected classes
    for i, c in enumerate(highlight_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
        roc_auc = auc(fpr, tpr)
        label_prefix = "⬇" if c in bottom5 else "⬆"
        axes[0].plot(fpr, tpr, color=colors[i], linewidth=1.5,
                     label=f"{label_prefix} {GTSRB_NAMES[c]} (AUC={roc_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — Top-5 (⬆) & Bottom-5 (⬇) Classes")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[0].grid(True, alpha=0.2)

    # PR curves for selected classes
    for i, c in enumerate(highlight_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_probs[:, c])
        ap = average_precision_score(y_bin[:, c], y_probs[:, c])
        label_prefix = "⬇" if c in bottom5 else "⬆"
        axes[1].plot(rec, prec, color=colors[i], linewidth=1.5,
                     label=f"{label_prefix} {GTSRB_NAMES[c]} (AP={ap:.3f})")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curves — Top-5 (⬆) & Bottom-5 (⬇) Classes")
    axes[1].legend(fontsize=6, loc="lower left")
    axes[1].grid(True, alpha=0.2)

    plt.suptitle("ROC and Precision-Recall Curves — CNN on GTSRB",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_pr_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ROC/PR curves saved.")


def generate_results_summary(cfg, test_metrics):
    """Generate a comprehensive results summary CSV comparing all models."""
    logs_dir = cfg["paths"]["logs"]
    results_dir = cfg["paths"]["results"]

    rows = []

    # CNN test metrics
    rows.append({
        "Model": "Custom CNN (3-block Residual)",
        "Split": "Test",
        "Accuracy": test_metrics["accuracy"],
        "Macro-F1": test_metrics["macro_f1"],
    })

    # SVM metrics
    svm_path = os.path.join(logs_dir, "svm_metrics.json")
    if os.path.exists(svm_path):
        with open(svm_path) as f:
            svm = json.load(f)
        for split in ["val", "test"]:
            if split in svm:
                rows.append({
                    "Model": "SVM (HOG Features)",
                    "Split": split.capitalize(),
                    "Accuracy": svm[split]["accuracy"],
                    "Macro-F1": svm[split]["macro_f1"],
                })

    # NLP metrics
    nlp_path = os.path.join(logs_dir, "nlp_metrics.json")
    if os.path.exists(nlp_path):
        with open(nlp_path) as f:
            nlp = json.load(f)
        rows.append({
            "Model": "TextCNN (NLP Sign Descriptions)",
            "Split": "Val",
            "Accuracy": nlp.get("val_accuracy", "N/A"),
            "Macro-F1": nlp.get("val_macro_f1", "N/A"),
        })

    # RL metrics
    rl_path = os.path.join(logs_dir, "rl_training_log.json")
    if os.path.exists(rl_path):
        with open(rl_path) as f:
            rl = json.load(f)
        summary = rl.get("summary", {})
        rows.append({
            "Model": "Q-Learning RL Agent",
            "Split": f"Multi-seed ({rl.get('num_seeds', 'N/A')} seeds)",
            "Accuracy": f"{summary.get('mean_success_rate', 'N/A')} "
                        f"± {summary.get('std_success_rate', 'N/A')}",
            "Macro-F1": f"Reward: {summary.get('mean_reward', 'N/A')} "
                        f"± {summary.get('std_reward', 'N/A')}",
        })

    # Ablation summaries
    for abl_name in ["augmentation", "lr", "depth"]:
        abl_path = os.path.join(logs_dir, f"ablation_{abl_name}.json")
        if os.path.exists(abl_path):
            with open(abl_path) as f:
                abl = json.load(f)
            for variant, metrics in abl.items():
                rows.append({
                    "Model": f"Ablation ({abl_name}): {variant}",
                    "Split": "Val",
                    "Accuracy": metrics.get("best_val_acc", "N/A"),
                    "Macro-F1": metrics.get("best_val_f1", "N/A"),
                })

    # Save CSV
    csv_path = os.path.join(results_dir, "results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Split",
                                               "Accuracy", "Macro-F1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results summary saved → {csv_path}")

    # Also print table
    print("\n  " + "=" * 72)
    print(f"  {'Model':<40s} {'Split':<12s} {'Accuracy':<12s} {'Macro-F1':<12s}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['Model']:<40s} {r['Split']:<12s} "
              f"{str(r['Accuracy']):<12s} {str(r['Macro-F1']):<12s}")
    print("  " + "=" * 72)

    return rows


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
    preds, labels, probs = get_predictions(model, test_loader, device)

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

    # ROC/PR curves
    print("\nGenerating ROC and PR curves ...")
    plot_roc_pr_curves(labels, probs, results_dir, num_classes)

    # Comprehensive results summary
    print("\nGenerating results summary table ...")
    generate_results_summary(cfg, test_metrics)

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
