# Traffic Sign Classifier — Project Proposal (DRAFT)

> **Status:** Pending approval. Details are proposed and subject to change.

A Python-based model for classifying road traffic signs. This project aims to build a reproducible, well-documented pipeline covering data ingestion, preprocessing, model training, evaluation, and inference — with planned extensions into NLP-assisted labeling and reinforcement learning.

---

## Table of Contents

- [Overview](#overview)
- [MVP](#mvp)
- [Planned Components](#planned-components)
- [Proposed Environment](#proposed-environment)
- [Data Download](#data-download)
- [Project Structure](#project-structure)
- [Project Board Tasks](#project-board-tasks)
- [Notes](#notes)

---

## Overview

This classifier will be trained to identify common traffic signs from image input. The proposed base dataset is the **TO-BE-DECIDED**.

The project is scoped in two stages: an **MVP** delivering a working CNN classifier, followed by planned **CNN, NLP, and RL extensions** that build on top of it.

---

## MVP

## Minimum Viable Product (MVP)

The MVP is a fully functional, end-to-end Traffic Sign Classification system that can be trained, evaluated, and used for inference through a reproducible pipeline.

---

### Dataset

The system uses the German Traffic Sign Recognition Benchmark (GTSRB), a widely used dataset for traffic sign classification.

- ~39,000 training images  
- ~12,000 test images  
- 43 traffic sign classes  

The dataset is automatically downloaded using `torchvision` via `data/get_data.py`, requiring no external API keys or manual setup. Data is stored locally in the `data/` directory.

---

### MVP Capabilities

#### 1. CNN-Based Traffic Sign Classifier
- A custom Convolutional Neural Network (`TrafficSignCNN`) implemented in PyTorch  
- Trained on GTSRB for multi-class traffic sign classification  
- Supports configurable architecture and hyperparameters via `configs/config.yaml`  

---

#### 2. Baseline Model (SVM + HOG)
- Traditional machine learning baseline using:
  - Histogram of Oriented Gradients (HOG)
  - Support Vector Machine (SVM)  
- Enables comparison between classical and deep learning approaches  

---

#### 3. Evaluation and Metrics
- Accuracy and Macro F1-score  
- Per-class classification performance  
- Confusion matrix visualization  
- Metrics and logs saved to:
  - `experiments/logs/`
  - `experiments/results/`  

---

#### 4. Reinforcement Learning Component
- A Q-learning–based RL agent (`rl_agent.py`)  
- Simulates decision-making in a grid environment using reward-based learning  
- Includes configurable parameters such as exploration rate, reward shaping, and multi-seed evaluation  

---

#### 5. NLP Component
- A lightweight NLP module (`nlp_component.py`)  
- Demonstrates text-based processing within the system  
- Serves as an auxiliary component to satisfy NLP integration requirements  

---

#### 6. Single Image Inference
- A command-line script (`predict.py`) for predicting the class of a single traffic sign image  


## Planned Components

### CNN — Core Classifier
The primary model. A convolutional neural network trained to map traffic sign images.

- **Baseline:** functioning CNN
- **Input:** RGB images resized to 32×32 or 64×64
- **Output:** Softmax probability distribution over sign classes
- **Key challenges:** Class imbalance (some sign types are rare), lighting and angle variation, small image resolution

### NLP — Label Assistance & Description Generation
An NLP component to assist with human-readable labeling and dataset annotation quality.

- **Proposed use:** Given a predicted class ID (e.g., `class_14`), generate a natural language description of the sign and its road meaning (e.g., *"Stop sign — vehicle must come to a full stop"*)
- **Approach:** A lightweight text generation or retrieval module mapping class labels to structured sign descriptions
- **Extended use:** Flagging ambiguous or low-confidence predictions with a plain-language explanation for human review
- **Tooling:** `transformers` (HuggingFace), or a rules-based mapping as a simpler first pass

### RL — Adaptive Inference Agent
A reinforcement learning layer on top of the trained CNN, proposed for a later date.

- **Proposed use:** An RL agent that learns when to defer a low-confidence prediction for human review vs. accept it, optimising for a reward signal balancing accuracy and throughput
- **Approach:** Frame inference as a sequential decision problem — the agent observes the CNN's confidence scores and decides: *accept*, *request second opinion*, or *flag for review*
- **Algorithm:** Proposed starting point is a simple policy gradient or Q-learning approach over the confidence action space
- **Why this matters:** In safety-critical contexts (e.g., autonomous driving), the cost of a wrong prediction is not uniform — RL can encode that asymmetry

---

## Proposed Environment

Dependencies are managed via **pip** using a `requirements.txt` file.

### `requirements.txt` (proposed)

```
python==3.10
torch>=2.0
torchvision>=0.15
opencv-python>=4.7
numpy>=1.24
pandas>=2.0
scikit-learn>=1.2
matplotlib>=3.7
albumentations>=1.3
tqdm>=4.65
pyyaml>=6.0
torch-summary
transformers>=4.30       # NLP component
gymnasium>=0.29          # RL component
```

### Setup (proposed commands)

```bash
# Clone the repository
git clone https://github.com/<org>/traffic-sign-classifier.git
cd traffic-sign-classifier

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

> **Note:** `transformers` and `gymnasium` are listed for proposed NLP and RL components. They can be deferred to a separate `requirements-extensions.txt` if the MVP scope is preferred for initial install.

---

## Data Download

A download script is provided as a template. URLs and target paths are configurable, swap in any compatible dataset by editing the variables at the top of the script.

### `scripts/download_data.py` (template)

```python
"""
Data Download Script — Traffic Sign Classifier
-----------------------------------------------
TEMPLATE: Replace the placeholder values below with your dataset source.
Run: python scripts/download_data.py
"""

import os
import urllib.request
import zipfile

# --- CONFIGURE THESE ---
DATASET_URL = "https://example.com/path/to/dataset.zip"   # TODO: replace with real URL
DATASET_ZIP = "data/raw/dataset.zip"
EXTRACT_DIR = "data/raw/"
TRAIN_DIR   = "data/train/"
TEST_DIR    = "data/test/"
# -----------------------

def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")

def extract(zip_path: str, extract_to: str) -> None:
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def organize() -> None:
    """
    TODO: Add logic to move/rename extracted files into
    data/train/<class_label>/ and data/test/<class_label>/
    folder structure expected by the DataLoader.
    """
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    print("Organize step: not yet implemented — see TODO above.")

if __name__ == "__main__":
    download(DATASET_URL, DATASET_ZIP)
    extract(DATASET_ZIP, EXTRACT_DIR)
    organize()
    print("Done. Verify structure under data/train/ and data/test/")
```

### Expected data structure after running the script

```
data/
├── raw/                  # original downloaded archive
├── train/
│   ├── 00_stop/
│   ├── 01_yield/
│   ├── 02_speed_limit_30/
│   └── ...
└── test/
    ├── 00_stop/
    └── ...
```

---

## Project Structure

The following layout is proposed and may be adjusted.

```
traffic-sign-classifier/
├── data/                      # gitignored — populated by download script
├── notebooks/                 # exploratory analysis and experiments
├── scripts/
│   └── download_data.py
├── src/
│   ├── dataset.py             # PyTorch Dataset and DataLoader
│   ├── model.py               # CNN architecture definition
│   ├── train.py               # training loop
│   ├── evaluate.py            # metrics and confusion matrix
│   ├── predict.py             # single-image inference (MVP deliverable)
│   ├── nlp/
│   │   └── label_helper.py    # NLP label descriptions and flagging
│   └── rl/
│       └── inference_agent.py # RL agent for adaptive inference decisions
├── configs/
│   └── config.yaml            # hyperparameters and paths
├── tests/                     # unit tests (proposed)
├── requirements.txt
└── README.md
```

---

## Project Board Tasks

The following tasks are proposed for the project board, grouped by phases. Final scope and assignments are pending and subject to change.

### Phase 1 — Setup & Data
- [ ] Finalize dataset choice
- [ ] Complete data download script with real URLs and organize logic
- [ ] Perform exploratory data analysis (class distribution, image quality)
- [ ] Define train/validation/test split strategy
- [ ] Implement PyTorch `Dataset` and `DataLoader` classes

### Phase 2 — CNN (MVP)
- [ ] Implement baseline CNN architecture
- [ ] Define augmentation pipeline (flips, brightness, rotation, noise)
- [ ] Write training loop with validation checkpointing
- [ ] Address class imbalance (oversampling or weighted loss)
- [ ] Verify MVP deliverables: inference script, metrics output, reproducible setup

### Phase 3 — NLP Component
- [ ] Map class IDs to plain-language sign descriptions
- [ ] Implement `label_helper.py` with description lookup and low-confidence flagging
- [ ] Evaluate NLP output quality against a baseline rules-based mapping
- [ ] Integrate NLP descriptions into inference output

### Phase 4 — RL Component
- [ ] Define the inference decision environment (states, actions, reward function)
- [ ] Implement a baseline RL agent using `gymnasium`
- [ ] Train agent against CNN confidence scores on validation set
- [ ] Evaluate agent vs. fixed confidence threshold baseline

### Phase 5 — Evaluation & Delivery
- [ ] Compute per-class accuracy, precision, recall, F1 across all components
- [ ] Generate and visualize confusion matrix
- [ ] Write unit tests for data pipeline, CNN output shapes, and NLP mapping
- [ ] Finalize `requirements.txt` and setup instructions
- [ ] Complete and review final README
- [ ] Package model weights and upload final release

---

## Notes

- The **MVP** is intended to be approvable and deliverable independently of the NLP and RL extensions.
- `transformers` and `gymnasium` dependencies can be split into a separate install step if one prefers a different MVP environment.
- The data download script is intentionally a template — the real dataset URL and organization logic will be filled in once a dataset is decided.

---

*DRAFT PREPARED FOR REVIEW, SUBJECT TO CHANGE. NOT FINAL*
