# data/get_data.py
"""
Downloads Kaggle dataset for traffic sign classification and unpacks it.

Before running:
 1. Install kaggle: pip install kaggle
 2. Save kaggle.json to ~/.kaggle/kaggle.json
"""

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASETS = [
    "ahemateja19bec1025/traffic-sign-dataset-classification",
    "valentynsichkar/traffic-signs-preprocessed",
    "tuanai/traffic-signs-dataset"
]

RAW_DIR = "data/raw"

def download_kaggle_dataset(dataset, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading {dataset} …")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=download_dir, unzip=False)
    print(f"Downloaded to {download_dir}")

def unzip_files(download_dir):
    for fname in os.listdir(download_dir):
        if fname.endswith(".zip"):
            fpath = os.path.join(download_dir, fname)
            print(f"Extracting {fname} …")
            with zipfile.ZipFile(fpath, "r") as zip_ref:
                zip_ref.extractall(download_dir)
            print("Done extracting.")

if __name__ == "__main__":
    for ds in DATASETS:
        download_kaggle_dataset(ds, RAW_DIR)
    unzip_files(RAW_DIR)
    print("All datasets downloaded and extracted!")
