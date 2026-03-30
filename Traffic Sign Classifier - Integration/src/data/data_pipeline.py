"""
Traffic Sign Classifier — Data Pipeline
========================================
Downloads GTSRB via torchvision, applies transforms, creates
stratified train / val / test DataLoaders.

Run standalone to verify:
    python src/data_pipeline.py
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(image_size: int, train: bool = True):
    """Return torchvision transform pipeline."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3401, 0.3120, 0.3212],
                                 std=[0.2725, 0.2609, 0.2669]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3401, 0.3120, 0.3212],
                                 std=[0.2725, 0.2609, 0.2669]),
        ])


def get_dataloaders(cfg: dict):
    """
    Download GTSRB and return train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, num_classes
    """
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    img_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    val_split = data_cfg["val_split"]
    num_workers = data_cfg.get("num_workers", 2)

    # Download datasets
    train_dataset = datasets.GTSRB(
        root=root, split="train", download=True,
        transform=get_transforms(img_size, train=True),
    )
    test_dataset = datasets.GTSRB(
        root=root, split="test", download=True,
        transform=get_transforms(img_size, train=False),
    )

    # Stratified train / val split
    targets = [s[1] for s in train_dataset._samples]
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=targets,
        random_state=cfg["seed"],
    )

    # Validation subset uses eval transforms (no augmentation)
    val_dataset = datasets.GTSRB(
        root=root, split="train", download=False,
        transform=get_transforms(img_size, train=False),
    )

    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    num_classes = 43  # GTSRB has 43 classes

    return train_loader, val_loader, test_loader, num_classes


# ------------------------------------------------------------------
# Quick verification
# ------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print("Loading GTSRB dataset ...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(cfg)
    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Val   samples : {len(val_loader.dataset)}")
    print(f"  Test  samples : {len(test_loader.dataset)}")
    print(f"  Num classes   : {num_classes}")
    # Verify a single batch
    imgs, labels = next(iter(train_loader))
    print(f"  Batch shape   : {imgs.shape}")
    print(f"  Label range   : {labels.min().item()} – {labels.max().item()}")
    print("Data pipeline OK ✓")