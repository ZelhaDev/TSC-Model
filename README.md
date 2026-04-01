# Traffic Sign Classifier

> **Status:** ✅ Working — All core components (CNN, NLP, RL) are trained and functional.

A Python-based pipeline for classifying road traffic signs, built on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset (43 classes). The project covers data ingestion, preprocessing, model training, evaluation, and inference — with integrated NLP-assisted labeling and a reinforcement learning grid-world agent.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Components](#components)
- [Environment & Setup](#environment--setup)
- [How to Reproduce](#how-to-reproduce)
- [Project Structure](#project-structure)
- [Project Board](#project-board)
- [Notes](#notes)

---

## Overview

This classifier identifies common traffic signs from image input using the **GTSRB** dataset — a widely used benchmark containing over 50,000 images across 43 sign classes under varying lighting, angle, and resolution conditions.

The project is composed of three integrated components:

| Component | Model | Status |
|-----------|-------|--------|
| **CNN** | Custom Residual CNN (from scratch) | ✅ Trained & evaluated |
| **NLP** | TextCNN on sign descriptions | ✅ Trained & evaluated |
| **RL** | Q-Learning Grid World Agent | ✅ Trained & evaluated |
| **SVM** | HOG + RBF SVM baseline | ✅ Trained & evaluated |

---

## Key Results

| Model | Split | Accuracy | Macro-F1 |
|-------|-------|----------|----------|
| Custom CNN (3-block Residual) | Test | 41.72% | 0.222 |
| SVM (HOG Features) | Test | 81.97% | 0.7515 |
| SVM (HOG Features) | Val | 91.44% | 0.9006 |
| TextCNN (NLP Sign Descriptions) | Val | 75.36% | 0.7423 |

**RL Agent:** Q-Learning agent successfully learns to navigate a 5×5 grid world with traffic-sign reward modifiers across 3 seeds with convergence confirmed via learning curves.

> **Note:** The CNN was trained with 30 epochs, cosine LR scheduling, and early stopping. Full evaluation includes confusion matrices, ROC/PR curves, ablation studies, error analysis, and Grad-CAM visualisations — all saved under `experiments/results/`.

---

## Components

### CNN — Core Classifier ✅

A custom convolutional neural network built **from scratch** with residual skip connections, Batch Normalization, and Global Average Pooling.

- **Architecture:** Configurable-depth residual blocks (2, 3, or 4 blocks). Default is 3 blocks with channel progression `[32 → 64 → 128]`.
- **Input:** RGB images resized to 32×32
- **Output:** Softmax probability distribution over 43 GTSRB classes
- **Features:**
  - Residual skip connections with 1×1 projection shortcuts
  - BatchNorm + ReLU activation
  - MaxPool2d downsampling per block
  - Dropout2d regularisation
  - Global Average Pooling → FC classifier (512 → 43)
  - Grad-CAM support via `get_feature_layer()`
- **Training:** Adam optimizer, cosine annealing LR (lr=0.0005), weight decay=1e-4, early stopping (patience=7)
- **Checkpoint:** `experiments/logs/best_model.pth`

### NLP — Label Assistance & Description Generation ✅

A TextCNN model that classifies structured sign descriptions, mapping class IDs to human-readable labels.

- **Use:** Given a predicted class ID (e.g., `class_14`), generates a natural language description of the sign and its road meaning (e.g., *"Stop sign — vehicle must come to a full stop"*)
- **Architecture:** TextCNN with embedding → multi-kernel convolution → max-pool → FC
- **Config:** Embedding=64, filters=32, kernel sizes=[2, 3, 4], dropout=0.3
- **Result:** 75.36% validation accuracy, 0.7423 macro-F1

### RL — Traffic Sign Grid World Agent ✅

A Q-learning agent that navigates a 5×5 grid world where traffic signs modify the reward signal. The CNN-RL bridge maps CNN predictions to grid-cell sign placements.

- **Environment:** 5×5 grid, agent starts at (0,0), goal at (4,4). Traffic signs are placed on cells with reward modifiers:
  - `stop` → −5.0 | `no_entry` → −8.0 | `yield` → −1.0
  - `speed_limit` → +2.0 | `priority` → +1.5
- **Agent:** Tabular Q-learning with ε-greedy exploration (ε decays from 1.0 → 0.05)
- **CNN Integration:** Loads the trained CNN checkpoint to classify representative sign images and maps GTSRB class predictions to grid sign types
- **Evaluation:** Multi-seed training (3 seeds) with variance reporting and confidence-band learning curves
- **Hyperparameters:** α=0.1, γ=0.95, 500 episodes, max 50 steps/episode
- **Output:** Learning curves with ±1 std bands saved to `experiments/results/rl_learning_curve.png`

### SVM — Baseline Classifier ✅

A traditional ML baseline using HOG features + RBF-kernel SVM.

- **Features:** Histogram of Oriented Gradients (HOG)
- **Kernel:** RBF (C=10.0)
- **Result:** 81.97% test accuracy, 0.7515 macro-F1

---

## Environment & Setup

**Python 3.10+** is required. Dependencies are managed via **pip**.

### `requirements.txt`

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.2
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
Pillow>=9.0
tqdm>=4.65
pyyaml>=6.0
gymnasium>=0.29
scikit-image>=0.20
opencv-python>=4.7
```

### Setup

```bash
# Clone the repository
git clone https://github.com/<org>/TSC-Model.git
cd TSC-Model

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

---

## How to Reproduce

A single script reproduces all results end-to-end:

```bash
bash run.sh            # full pipeline (includes ablations)
bash run.sh --quick    # skip ablations for faster run
```

### Pipeline Steps

| Step | Command | Description |
|------|---------|-------------|
| 0 | `python data/get_data.py` | Download GTSRB dataset via torchvision |
| 1 | `pip install -r requirements.txt` | Install dependencies |
| 2 | `python src/nlp_component.py` | Train NLP TextCNN component |
| 3 | `python src/rl_agent.py` | Train RL agent (multi-seed Q-learning) |
| 4 | `python src/train.py` | Train CNN and SVM models |
| 5 | `python src/eval.py` | Evaluate: test metrics, confusion matrix, ROC/PR |
| 6 | `python src/ablation_runner.py` | Run ablation studies (augmentation, LR, depth) |
| 7 | `python src/error_analysis.py` & `python src/grad_cam.py` | Error analysis & Grad-CAM visualisations |

### Output Locations

- **Metric logs:** `experiments/logs/` — JSON files with training/test metrics, RL logs
- **Plots & results:** `experiments/results/` — confusion matrix, learning curves, ROC/PR curves, RL plots
- **Checkpoints:** `experiments/logs/best_model.pth`

---

## Project Structure

```
TSC-Model/
├── configs/
│   └── config.yaml                # hyperparameters and paths
├── data/
│   └── get_data.py                # GTSRB download script (torchvision)
├── experiments/
│   ├── logs/                      # training logs, metrics, checkpoints
│   │   ├── best_model.pth         # trained CNN checkpoint
│   │   ├── training_metrics.json
│   │   ├── test_metrics.json
│   │   ├── svm_metrics.json
│   │   ├── nlp_metrics.json
│   │   └── rl_training_log.json
│   └── results/                   # generated plots and summaries
│       ├── confusion_matrix.png
│       ├── learning_curves.png
│       ├── roc_pr_curves.png
│       ├── rl_learning_curve.png
│       └── results_summary.csv
├── notebooks/                     # exploratory analysis
├── src/
│   ├── models/
│   │   └── cnn.py                 # Custom Residual CNN architecture
│   ├── data_pipeline.py           # PyTorch Dataset, DataLoader, transforms
│   ├── train.py                   # CNN + SVM training loop
│   ├── eval.py                    # evaluation pipeline (metrics, plots)
│   ├── nlp_component.py           # TextCNN for sign descriptions
│   ├── rl_agent.py                # Q-Learning grid world + CNN integration
│   ├── ablation_runner.py         # ablation studies (augmentation, LR, depth)
│   ├── error_analysis.py          # per-class error breakdown
│   └── grad_cam.py                # Grad-CAM visualisations
├── RL_Prototype.py                # early Q-learning prototype
├── run.sh                         # one-command reproduce script
├── requirements.txt
└── README.md
```

---

## Project Board

### Phase 1 — Setup & Data
- [x] Finalize dataset choice → **GTSRB (43 classes)**
- [x] Implement data download script (`data/get_data.py` via torchvision)
- [x] Perform exploratory data analysis (class distribution, image quality)
- [x] Define train/validation/test split strategy (80/20 stratified)
- [x] Implement PyTorch `Dataset` and `DataLoader` classes

### Phase 2 — CNN (Core Model)
- [x] Implement custom CNN architecture with residual blocks
- [x] Define augmentation pipeline (transforms in data pipeline)
- [x] Write training loop with validation checkpointing and early stopping
- [x] Address class imbalance (weighted loss)
- [x] Verify deliverables: inference, metrics output, reproducible setup

### Phase 3 — NLP Component
- [x] Map class IDs to plain-language sign descriptions
- [x] Implement TextCNN for sign description classification
- [x] Evaluate NLP output quality (75.36% val accuracy)
- [x] Integrate NLP descriptions into inference output

### Phase 4 — RL Component
- [x] Define the grid-world environment (states, actions, reward function)
- [x] Implement Q-Learning agent with ε-greedy exploration
- [x] Build CNN → RL bridge (CNN predictions → grid sign placements)
- [x] Multi-seed training (3 seeds) with variance reporting
- [x] Plot learning curves with confidence bands

### Phase 5 — Evaluation & Delivery
- [x] Compute per-class accuracy, precision, recall, F1 across all components
- [x] Generate and visualize confusion matrix, ROC/PR curves
- [x] Run ablation studies (augmentation, learning rate, CNN depth)
- [x] Run error analysis and Grad-CAM visualisations
- [x] Finalize `requirements.txt` and setup instructions
- [x] Complete `run.sh` one-command reproduce script
- [ ] Package model weights and upload final release

---

## Notes

- The **GTSRB dataset** is automatically downloaded via `torchvision.datasets.GTSRB` when running the pipeline.
- All configuration (hyperparameters, paths, ablation variants) is centralized in `configs/config.yaml`.
- The RL agent works both standalone (default sign placements) and with CNN integration (loads checkpoint to classify signs for the grid).
- The SVM baseline (HOG + RBF) serves as a traditional ML comparison point against the deep learning CNN.
- Ablation studies cover three axes: data augmentation (on/off), learning rate (0.001, 0.0005, 0.0001), and CNN depth (2, 3, 4 blocks).

---

*Last updated: April 2026*