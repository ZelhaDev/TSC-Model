"""
Traffic Sign Classifier — Ablation Studies
============================================
Runs ≥2 ablation experiments to compare model variants:
  A1. Augmentation ON vs OFF
  A2. Learning rate comparison (0.001, 0.0005, 0.0001)
  A3. Architecture depth (2-block vs 3-block vs 4-block)

Each ablation trains with the same seed and reduced epochs (configurable)
for fair comparison, then produces comparison plots.

Run:
    python src/ablation_runner.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline import load_config, get_dataloaders, get_transforms
from models.cnn import TrafficSignCNN
from train import set_seed, train_cnn, train_one_epoch, evaluate

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# ==============================================================
# Helpers
# ==============================================================

def get_dataloaders_no_aug(cfg):
    """Get dataloaders WITHOUT augmentation (for ablation A1)."""
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    img_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    val_split = data_cfg["val_split"]
    num_workers = data_cfg.get("num_workers", 2)

    # Both train and val use eval transforms (no augmentation)
    eval_tf = get_transforms(img_size, train=False)

    train_dataset = datasets.GTSRB(
        root=root, split="train", download=True, transform=eval_tf,
    )
    test_dataset = datasets.GTSRB(
        root=root, split="test", download=True, transform=eval_tf,
    )

    targets = [s[1] for s in train_dataset._samples]
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=targets,
        random_state=cfg["seed"],
    )

    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


def plot_ablation_comparison(results, xlabel, title, save_path, metric="val_f1"):
    """Plot comparison bar chart + learning curves for an ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: final metric
    names = list(results.keys())
    final_vals = [h[metric][-1] for h in results.values()]
    best_vals = [max(h[metric]) for h in results.values()]

    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, final_vals, width, label="Final", color="#4C72B0")
    axes[0].bar(x + width/2, best_vals, width, label="Best", color="#55A868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Macro-F1" if metric == "val_f1" else metric)
    axes[0].set_title(f"{title} — Final vs Best {metric}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Learning curves
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for i, (name, h) in enumerate(results.items()):
        epochs = range(1, len(h[metric]) + 1)
        axes[1].plot(epochs, h[metric], "o-", color=colors[i % len(colors)],
                     label=name, markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1" if metric == "val_f1" else metric)
    axes[1].set_title(f"{title} — Validation Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Ablation: {title}", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ablation plot saved → {save_path}")


# ==============================================================
# Ablation A1: Augmentation ON vs OFF
# ==============================================================

def ablation_augmentation(cfg, abl_epochs):
    """Compare training with and without data augmentation."""
    print("\n" + "=" * 60)
    print("  ABLATION A1: Data Augmentation ON vs OFF")
    print("=" * 60)

    results = {}

    # --- With augmentation (default) ---
    set_seed(cfg["seed"])
    train_loader, val_loader, _, _ = get_dataloaders(cfg)
    _, hist_aug = train_cnn(cfg, train_loader, val_loader,
                            epochs_override=abl_epochs, tag="abl_aug_on")
    results["Aug ON"] = hist_aug

    # --- Without augmentation ---
    set_seed(cfg["seed"])
    train_loader_no, val_loader_no = get_dataloaders_no_aug(cfg)
    _, hist_noaug = train_cnn(cfg, train_loader_no, val_loader_no,
                              epochs_override=abl_epochs, tag="abl_aug_off")
    results["Aug OFF"] = hist_noaug

    save_path = os.path.join(cfg["paths"]["results"], "ablation_augmentation.png")
    plot_ablation_comparison(results, "Augmentation", "Data Augmentation",
                            save_path)

    # Save numeric results
    summary = {}
    for name, h in results.items():
        summary[name] = {
            "best_val_f1": round(max(h["val_f1"]), 4),
            "final_val_f1": round(h["val_f1"][-1], 4),
            "best_val_acc": round(max(h["val_acc"]), 4),
        }
    log_path = os.path.join(cfg["paths"]["logs"], "ablation_augmentation.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


# ==============================================================
# Ablation A2: Learning Rate Comparison
# ==============================================================

def ablation_learning_rate(cfg, abl_epochs):
    """Compare different learning rates."""
    print("\n" + "=" * 60)
    print("  ABLATION A2: Learning Rate Comparison")
    print("=" * 60)

    variants = cfg["ablations"]["learning_rate"]["variants"]
    results = {}

    set_seed(cfg["seed"])
    train_loader, val_loader, _, _ = get_dataloaders(cfg)

    for lr in variants:
        set_seed(cfg["seed"])
        label = f"LR={lr}"
        _, hist = train_cnn(cfg, train_loader, val_loader,
                            lr_override=lr, epochs_override=abl_epochs,
                            tag=f"abl_lr_{lr}")
        results[label] = hist

    save_path = os.path.join(cfg["paths"]["results"], "ablation_lr.png")
    plot_ablation_comparison(results, "Learning Rate",
                            "Learning Rate Schedule", save_path)

    summary = {}
    for name, h in results.items():
        summary[name] = {
            "best_val_f1": round(max(h["val_f1"]), 4),
            "final_val_f1": round(h["val_f1"][-1], 4),
            "best_val_acc": round(max(h["val_acc"]), 4),
        }
    log_path = os.path.join(cfg["paths"]["logs"], "ablation_lr.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


# ==============================================================
# Ablation A3: Architecture Depth
# ==============================================================

def ablation_depth(cfg, abl_epochs):
    """Compare CNN with different number of residual blocks."""
    print("\n" + "=" * 60)
    print("  ABLATION A3: Architecture Depth (Num Blocks)")
    print("=" * 60)

    variants = cfg["ablations"]["depth"]["variants"]
    results = {}

    set_seed(cfg["seed"])
    train_loader, val_loader, _, _ = get_dataloaders(cfg)

    for nb in variants:
        set_seed(cfg["seed"])
        label = f"{nb}-block"
        _, hist = train_cnn(cfg, train_loader, val_loader,
                            num_blocks=nb, epochs_override=abl_epochs,
                            tag=f"abl_depth_{nb}")
        results[label] = hist

    save_path = os.path.join(cfg["paths"]["results"], "ablation_depth.png")
    plot_ablation_comparison(results, "Architecture Depth",
                            "CNN Depth (Residual Blocks)", save_path)

    summary = {}
    for name, h in results.items():
        summary[name] = {
            "best_val_f1": round(max(h["val_f1"]), 4),
            "final_val_f1": round(h["val_f1"][-1], 4),
            "best_val_acc": round(max(h["val_acc"]), 4),
        }
    log_path = os.path.join(cfg["paths"]["logs"], "ablation_depth.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    return results


# ==============================================================
# Main
# ==============================================================

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    abl_epochs = cfg.get("ablations", {}).get("epochs", 10)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)

    print(f"\nAblation budget: {abl_epochs} epochs per variant\n")

    # Run all three ablations
    ablation_augmentation(cfg, abl_epochs)
    ablation_learning_rate(cfg, abl_epochs)
    ablation_depth(cfg, abl_epochs)

    print("\n✓ All ablation studies complete.")
    print("  Plots saved to experiments/results/ablation_*.png")
    print("  Logs  saved to experiments/logs/ablation_*.json")


if __name__ == "__main__":
    main()
