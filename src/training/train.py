"""
Traffic Sign Classifier — Training Pipeline
=============================================
Trains:
  1. SVM baseline on HOG features
  2. Custom CNN (TrafficSignCNN)

Logs metrics to experiments/logs/.

Run:
    python src/train.py
"""

import os
import sys
import json
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from skimage.feature import hog

# Local imports
from src.data.data_pipeline import load_config, get_dataloaders
from src.models.cnn import TrafficSignCNN


# ==============================================================
# Utilities
# ==============================================================

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_hog_features(loader, max_samples: int = 10000):
    """Extract HOG features from a DataLoader for SVM."""
    features, labels = [], []
    count = 0
    for imgs, lbls in loader:
        for img, lbl in zip(imgs, lbls):
            if count >= max_samples:
                return np.array(features), np.array(labels)
            # Convert to grayscale numpy
            gray = img.mean(dim=0).numpy()
            fd = hog(gray, orientations=9, pixels_per_cell=(4, 4),
                     cells_per_block=(2, 2), feature_vector=True)
            features.append(fd)
            labels.append(lbl.item())
            count += 1
    return np.array(features), np.array(labels)


# ==============================================================
# SVM Baseline
# ==============================================================

def train_svm(cfg, train_loader, val_loader, test_loader):
    """Train an SVM on HOG features and log metrics."""
    print("\n" + "=" * 60)
    print("  SVM BASELINE (HOG Features)")
    print("=" * 60)

    svm_cfg = cfg["svm"]
    max_samples = svm_cfg.get("max_samples", 10000)

    print(f"Extracting HOG features (max {max_samples} samples) ...")
    X_train, y_train = extract_hog_features(train_loader, max_samples)
    X_val, y_val = extract_hog_features(val_loader, max_samples // 4)
    X_test, y_test = extract_hog_features(test_loader, max_samples)

    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
          f"Test: {X_test.shape[0]}")
    print(f"  HOG feature dim: {X_train.shape[1]}")

    print(f"Training SVM (C={svm_cfg['C']}, kernel={svm_cfg['kernel']}) ...")
    clf = SVC(C=svm_cfg["C"], kernel=svm_cfg["kernel"], random_state=cfg["seed"])
    clf.fit(X_train, y_train)

    # Evaluate
    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        results[split_name] = {"accuracy": round(acc, 4),
                               "macro_f1": round(f1, 4)}
        print(f"  {split_name.upper():5s} — Accuracy: {acc:.4f},  "
              f"Macro-F1: {f1:.4f}")

    # Save
    log_path = os.path.join(cfg["paths"]["logs"], "svm_metrics.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  SVM metrics saved → {log_path}")

    return results


# ==============================================================
# CNN Training
# ==============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss, accuracy, macro-F1."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / total, acc, f1


def train_cnn(cfg, train_loader, val_loader):
    """Train the custom CNN and return the best model + training history."""
    print("\n" + "=" * 60)
    print("  CNN TRAINING (TrafficSignCNN)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tcfg = cfg["training"]
    model = TrafficSignCNN(
        num_classes=cfg["cnn"]["num_classes"],
        dropout=cfg["cnn"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                           weight_decay=tcfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=tcfg["scheduler_step"],
                       gamma=tcfg["scheduler_gamma"])

    best_val_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{tcfg['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  ({elapsed:.1f}s)")

        # Early stopping / checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= tcfg["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch} "
                      f"(patience={tcfg['early_stopping_patience']})")
                break

    # Save training history
    log_path = os.path.join(cfg["paths"]["logs"], "training_metrics.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training metrics saved → {log_path}")
    print(f"  Best model saved       → {ckpt_path}")
    print(f"  Best val macro-F1      : {best_val_f1:.4f}")

    # Load best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    return model, history


# ==============================================================
# Main
# ==============================================================

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    print("Loading data ...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(cfg)

    # 1) SVM baseline
    train_svm(cfg, train_loader, val_loader, test_loader)

    # 2) CNN training
    model, history = train_cnn(cfg, train_loader, val_loader)

    print("\n✓ Training complete. Run src/eval.py for full test evaluation.")


if __name__ == "__main__":
    main()