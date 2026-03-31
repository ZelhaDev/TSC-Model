"""
Quick CNN Training Script
=========================
Downloads GTSRB dataset and trains TrafficSignCNN.
Saves the checkpoint to checkpoints/best_model.pth for use by main.py.

Run:
    python train_model.py
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.cnn import TrafficSignCNN
from src.data.data_pipeline import load_config

def main():
    # Load config
    cfg = load_config("configs/config.yaml")

    # Seeds
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]

    # Transforms
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3212],
                             std=[0.2725, 0.2609, 0.2669]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3401, 0.3120, 0.3212],
                             std=[0.2725, 0.2609, 0.2669]),
    ])

    # Download and load GTSRB
    data_root = cfg["data"]["root"]
    print("Downloading GTSRB dataset (this may take a few minutes)...")
    train_dataset = datasets.GTSRB(root=data_root, split="train",
                                    download=True, transform=train_tfm)
    val_dataset = datasets.GTSRB(root=data_root, split="train",
                                  download=False, transform=eval_tfm)

    # Stratified split
    targets = [s[1] for s in train_dataset._samples]
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg["data"]["val_split"],
        stratify=targets, random_state=seed)

    train_loader = DataLoader(Subset(train_dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx),
                            batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Model
    model = TrafficSignCNN(num_classes=43, dropout=cfg["cnn"]["dropout"]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: TrafficSignCNN ({params:,} params)")

    criterion = nn.CrossEntropyLoss()
    tcfg = cfg["training"]
    optimizer = optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                           weight_decay=tcfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=tcfg["scheduler_step"],
                       gamma=tcfg["scheduler_gamma"])

    # Training
    epochs = tcfg["epochs"]
    ckpt_path = os.path.join("checkpoints", "best_model.pth")
    os.makedirs("checkpoints", exist_ok=True)

    best_val_f1 = 0.0
    patience = 0
    print(f"\nTraining for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        train_loss /= total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, vcorrect, vtotal = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                vcorrect += (preds == labels).sum().item()
                vtotal += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= vtotal
        val_acc = vcorrect / vtotal
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        scheduler.step()
        elapsed = time.time() - t0

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = " << saved"
        else:
            patience += 1

        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}  ({elapsed:.1f}s){marker}")

        if patience >= tcfg["early_stopping_patience"]:
            print(f"  Early stopping (patience={tcfg['early_stopping_patience']})")
            break

    print(f"\nBest val F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved: {ckpt_path}")
    print("\nDone! You can now run: python -m src.main")

if __name__ == "__main__":
    main()
