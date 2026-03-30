"""
Traffic Sign Classifier — Error & Slice Analysis
===================================================
Produces detailed per-class breakdowns, failure-case galleries,
sign-category subgroup analysis, and top confusion-pair reports.

Run:
    python src/error_analysis.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)

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

# Sign category groupings for slice analysis
SIGN_CATEGORIES = {
    "Speed Limits": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "Prohibitory": [9, 10, 15, 16, 17],
    "Warning": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Mandatory": [33, 34, 35, 36, 37, 38, 39, 40],
    "Informational": [11, 12, 13, 14, 32, 41, 42],
}


# ==============================================================
# Get predictions
# ==============================================================

@torch.no_grad()
def get_predictions(model, loader, device):
    """Return predictions, labels, and raw images."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_images = []
    for imgs, labels in loader:
        imgs_dev = imgs.to(device)
        logits = model(imgs_dev)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(imgs.numpy())
    return (np.array(all_preds), np.array(all_labels),
            np.array(all_probs), np.array(all_images))


# ==============================================================
# Per-class analysis
# ==============================================================

def per_class_analysis(y_true, y_pred, save_dir):
    """Bar chart of per-class F1 scores, highlighting worst classes."""
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_acc = []
    for c in range(43):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append((y_pred[mask] == c).mean())
        else:
            per_class_acc.append(0.0)
    per_class_acc = np.array(per_class_acc)

    # Sort by F1 for readability
    order = np.argsort(per_class_f1)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#C44E52" if f < 0.7 else "#4C72B0" for f in per_class_f1[order]]
    y_pos = np.arange(43)
    ax.barh(y_pos, per_class_f1[order], color=colors, edgecolor="white",
            linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([GTSRB_NAMES[i] for i in order], fontsize=7)
    ax.set_xlabel("Macro-F1 Score")
    ax.set_title("Per-Class F1 Scores (Red = Below 0.70)")
    ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="Threshold 0.70")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_analysis_per_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Per-class F1 chart saved.")

    # Save numeric data
    per_class_data = {}
    for i in range(43):
        per_class_data[GTSRB_NAMES[i]] = {
            "class_id": int(i),
            "f1": round(float(per_class_f1[i]), 4),
            "accuracy": round(float(per_class_acc[i]), 4),
            "support": int((y_true == i).sum()),
        }
    return per_class_data


# ==============================================================
# Top confusion pairs
# ==============================================================

def top_confusion_pairs(y_true, y_pred, save_dir, top_k=15):
    """Find and visualize the most frequently confused class pairs."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # zero out correct predictions

    # Get top-k off-diagonal pairs
    pairs = []
    for i in range(43):
        for j in range(43):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    top_pairs = pairs[:top_k]

    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{GTSRB_NAMES[p[0]]} → {GTSRB_NAMES[p[1]]}" for p in top_pairs]
    counts = [p[2] for p in top_pairs]
    y_pos = np.arange(len(top_pairs))
    ax.barh(y_pos, counts, color="#DD8452", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Misclassification Count")
    ax.set_title(f"Top {top_k} Most Confused Class Pairs (True → Predicted)")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_analysis_top_confusions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Top confusion pairs chart saved.")

    return top_pairs


# ==============================================================
# Slice / subgroup analysis
# ==============================================================

def slice_analysis(y_true, y_pred, save_dir):
    """Compare performance across sign category groups."""
    slice_metrics = {}
    for cat_name, class_ids in SIGN_CATEGORIES.items():
        mask = np.isin(y_true, class_ids)
        if mask.sum() == 0:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        acc = accuracy_score(y_t, y_p)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        slice_metrics[cat_name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "support": int(mask.sum()),
            "classes": class_ids,
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(slice_metrics.keys())
    accs = [slice_metrics[n]["accuracy"] for n in names]
    f1s = [slice_metrics[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x, accs, width, color="#4C72B0", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Sign Category")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.2, axis="y")

    axes[1].bar(x, f1s, width, color="#55A868", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1 by Sign Category")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.2, axis="y")

    plt.suptitle("Slice Analysis — Performance by Sign Category",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "error_analysis_slices.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Slice analysis chart saved.")

    return slice_metrics


# ==============================================================
# Failure case gallery
# ==============================================================

def failure_gallery(y_true, y_pred, images, save_dir, num_cases=20):
    """Save a gallery of misclassified examples."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("  No misclassifications found!")
        return

    # Sample up to num_cases
    rng = np.random.RandomState(42)
    indices = rng.choice(wrong, size=min(num_cases, len(wrong)), replace=False)

    cols = 5
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    # De-normalize for display
    mean = np.array([0.3401, 0.3120, 0.3212]).reshape(3, 1, 1)
    std = np.array([0.2725, 0.2609, 0.2669]).reshape(3, 1, 1)

    for i, idx in enumerate(indices):
        img = images[idx] * std + mean  # de-normalize
        img = np.clip(img.transpose(1, 2, 0), 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {GTSRB_NAMES[y_true[idx]]}\n"
            f"Pred: {GTSRB_NAMES[y_pred[idx]]}",
            fontsize=7, color="red"
        )
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Failure Cases — Misclassified Test Samples",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "failure_cases.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Failure gallery saved ({len(indices)} cases).")


# ==============================================================
# Main
# ==============================================================

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  ERROR & SLICE ANALYSIS")
    print("=" * 60)

    print("Loading data ...")
    _, _, test_loader, num_classes = get_dataloaders(cfg)

    # Load best model
    ckpt = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    model = TrafficSignCNN(num_classes=num_classes,
                           dropout=cfg["cnn"]["dropout"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))
    print(f"Loaded model from {ckpt}")

    # Get all predictions
    y_pred, y_true, probs, images = get_predictions(model, test_loader, device)
    results_dir = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    # 1. Per-class analysis
    print("\n[1/4] Per-class F1 breakdown ...")
    per_class_data = per_class_analysis(y_true, y_pred, results_dir)

    # 2. Top confusion pairs
    print("[2/4] Top confusion pairs ...")
    top_pairs = top_confusion_pairs(y_true, y_pred, results_dir)

    # 3. Slice analysis
    print("[3/4] Slice / subgroup analysis ...")
    slice_data = slice_analysis(y_true, y_pred, results_dir)

    # 4. Failure gallery
    print("[4/4] Failure case gallery ...")
    failure_gallery(y_true, y_pred, images, results_dir)

    # Save full analysis JSON
    analysis_summary = {
        "overall": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "total_samples": int(len(y_true)),
            "misclassified": int((y_true != y_pred).sum()),
        },
        "per_class": per_class_data,
        "slices": slice_data,
        "top_confusion_pairs": [
            {"true": GTSRB_NAMES[p[0]], "predicted": GTSRB_NAMES[p[1]],
             "count": int(p[2])}
            for p in top_pairs[:10]
        ],
    }
    log_path = os.path.join(cfg["paths"]["logs"], "error_analysis.json")
    with open(log_path, "w") as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"\n  Full analysis saved → {log_path}")

    # Print worst-5 classes
    worst = sorted(per_class_data.items(), key=lambda x: x[1]["f1"])[:5]
    print(f"\n  Worst-5 classes by F1:")
    for name, d in worst:
        print(f"    {name:20s}  F1={d['f1']:.4f}  Acc={d['accuracy']:.4f}  "
              f"n={d['support']}")

    print("\n✓ Error analysis complete.")


if __name__ == "__main__":
    main()
