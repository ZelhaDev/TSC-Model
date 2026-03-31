"""
Traffic Sign Classifier — Dataset Download
===========================================
Downloads the German Traffic Sign Recognition Benchmark (GTSRB)
via torchvision. No Kaggle account or API key required.

Both the training split (~39,209 images) and test split (12,630 images)
are downloaded to the path specified in configs/config.yaml (data.root).

Run:
    python data/get_data.py

The download is skipped automatically if the data already exists.
"""

import os
import sys

# Allow running from either the project root or the data/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torchvision import datasets


def download_gtsrb(root: str = "data") -> None:
    """Download GTSRB train and test splits to `root` using torchvision."""
    print("=" * 55)
    print("  GTSRB Download")
    print("=" * 55)

    for split in ("train", "test"):
        print(f"\n  Checking {split} split ...")
        dataset = datasets.GTSRB(root=root, split=split, download=True)
        print(f"  ✓ {split.capitalize()} split ready — {len(dataset):,} images")

    print("\n  Download complete.")
    print(f"  Data stored in: {os.path.abspath(root)}/gtsrb/")
    print("\n  Run the full pipeline with:")
    print("    bash run.sh")


if __name__ == "__main__":
    # Read root from config if available, otherwise default to "data"
    root = "data"
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        root = cfg.get("data", {}).get("root", "data")
    except Exception:
        pass  # fall back to default

    download_gtsrb(root)
