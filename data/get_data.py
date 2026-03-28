import os
from kaggle.api.kaggle_api_extended import KaggleApi
import getpass

# Where to store raw datasets
RAW_DIR = "data/raw"

# List of Kaggle datasets to download
DATASETS = [
    "ahemateja19bec1025/traffic-sign-dataset-classification",
    "valentynsichkar/traffic-signs-preprocessed",
    "tuanai/traffic-signs-dataset"
]

def download_dataset(dataset, api):
    """Download and unzip a dataset if not already present."""
    dataset_name = dataset.split("/")[-1]
    path = os.path.join(RAW_DIR, dataset_name)

    if os.path.exists(path) and os.listdir(path):
        print(f"{dataset_name} already exists, skipping download...")
        return

    os.makedirs(path, exist_ok=True)

    print(f"Downloading {dataset}...")
    api.dataset_download_files(dataset, path=path, unzip=True)
    print(f"✅ Done: {path}\n")

if __name__ == "__main__":
    print("=== Kaggle API Setup ===")
    username = input("Enter your Kaggle username: ").strip()
    key = getpass.getpass("Enter your Kaggle API key (hidden): ").strip()

    # Temporarily set environment variables
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download all datasets
    for ds in DATASETS:
        download_dataset(ds, api)

    print("🎉 All datasets are ready in data/raw/!")
