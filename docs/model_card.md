# Model Card
## Traffic Sign Classifier (GTSRB) — Half_Exception, CS-304

**Version:** v0.1  
**Release:** https://github.com/ZelhaDev/TSC-Model/releases/tag/Releases  
**Date:** March 2026  
**Team:** Ahmad, Saeed · Aveena, Stephen · Predilla, Stanley · Siron, Carlo

---

## Model Overview

This project delivers three cooperating models trained on the German Traffic Sign Recognition Benchmark (GTSRB):

| Component | Architecture | Task |
|---|---|---|
| **CNN Classifier** | Custom 3-block Residual CNN | Image → 43-class traffic sign label |
| **TextCNN (NLP)** | 1-D CNN over word embeddings | Sign description text → 43-class label |
| **Q-Learning Agent (RL)** | Tabular Q-learning, ε-greedy | Grid-world navigation respecting sign semantics |

The primary model evaluated for deployment guidance is the **CNN Classifier**.

---

## Intended Use

### In-Scope
- Academic research on traffic sign recognition
- Benchmarking CNN architectures on GTSRB
- Educational demonstrations of multi-modal ML pipelines (CNN + NLP + RL)
- Prototyping explainability techniques (Grad-CAM) on sign imagery

### Out-of-Scope / Prohibited
- Integration into any production autonomous or semi-autonomous vehicle system
- Real-time traffic management or infrastructure control
- Any application where a misclassification could cause physical harm
- Deployment on sign standards outside the GTSRB / German convention without revalidation

---

## Training Data

| Property | Detail |
|---|---|
| **Dataset** | German Traffic Sign Recognition Benchmark (GTSRB) |
| **Source** | `torchvision.datasets.GTSRB` (auto-download to `data/gtsrb/`) |
| **Training images** | 39,209 (stratified 80/20 → ~31,367 train / ~7,842 val) |
| **Test images** | 12,630 |
| **Classes** | 43 traffic sign categories |
| **Input resolution** | Variable originals (25×25 → 266×232); resized to 32×32 for training |
| **Augmentations** | RandomRotation(±15°), ColorJitter (brightness/contrast/saturation ±0.2), RandomAffine (translate ±10%), ToTensor, Normalize |
| **Normalization** | Mean [0.3401, 0.3120, 0.3212], Std [0.2725, 0.2609, 0.2669] |

---

## Model Architecture

### CNN Classifier (`src/models/cnn.py`)

- **Backbone:** 3 × ResidualBlock (32 → 64 → 128 channels)
- **Each block:** Conv3×3 → BN → ReLU → Conv3×3 → BN + residual → MaxPool2d → Dropout2d
- **Head:** Global Average Pooling → Linear(128, 512) → ReLU → Dropout → Linear(512, 43)
- **Total parameters:** ~350K (3-block configuration)
- **Optimizer:** Adam, lr=0.0005, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR, T_max=30, eta_min=1e-6
- **Early stopping:** patience=7 epochs on val macro-F1
- **Training budget:** 30 epochs maximum

### TextCNN (`src/nlp_component.py`)

- Word-level vocabulary over 43 augmented sign descriptions (~688 tokens)
- Embedding dim: 64, filter sizes: {2, 3, 4}, 32 filters each
- Trained on synthetically augmented description corpus (43 × 16 = 688 samples; 80/20 split)

### Q-Learning Agent (`src/rl_agent.py`)

- State space: 25 (5×5 grid, flattened)
- Action space: 4 (up / down / left / right)
- Hyperparameters: α=0.1, γ=0.95, ε: 1.0 → 0.05 (decay 0.995/episode)
- Training: 500 episodes × 3 seeds

---

## Performance Metrics

### CNN Classifier

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation | ~best checkpoint | Tracked per epoch |
| **Test** | **81.97%** | **74.23%** |

### SVM Baseline (HOG features)

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation | 91.44% | 90.06% |

> **Note:** The SVM baseline uses HOG features extracted from the same 32×32 images and outperforms the CNN at this prototype stage. The CNN is architecturally positioned for further improvement with continued training. The SVM is not suitable for real-time embedded deployment due to inference speed constraints.

### TextCNN (NLP)

| Split | Accuracy | Macro-F1 |
|---|---|---|
| Validation | 81.36% | 75.15% |

> Trained on synthetic augmented data only; performance reflects classification of textual sign descriptions, not real images.

### Q-Learning Agent

| Metric | Result |
|---|---|
| Success rate (final 100 ep) | >95% (3-seed average) |
| Evaluation | Multi-seed (3 seeds), 500 episodes each |

---

## Performance by Sign Subgroup (Slice Analysis)

Based on GTSRB test set label distribution. CNN performance varies significantly across sign semantic categories:

| Category | Classes | Test Count | Notes |
|---|---|---|---|
| **Speed Limits** | 0–8 | ~4,590 | Dominant in dataset; likely highest accuracy |
| **Prohibitory** | 9, 10, 15, 16, 17 | ~1,560 | Medium representation |
| **Warning** | 18–31 | ~2,130 | Visually diverse; includes rare classes |
| **Mandatory** | 33–40 | ~1,680 | Generally well-represented |
| **Informational** | 11, 12, 13, 14, 32, 41, 42 | ~2,040 | Mixed; Stop/Yield are critical |

### High-Risk Underrepresented Classes (Test Set)

These classes have ≤60 test images — performance estimates carry high variance and should be interpreted cautiously:

| Class ID | Sign Name | Test Images |
|---|---|---|
| 0 | Speed limit 20 km/h | 60 |
| 19 | Dangerous curve left | 60 |
| 27 | Pedestrians | 60 |
| 32 | End of all limits | 60 |
| 37 | Go straight or left | 60 |
| 41 | End of no passing | 60 |

> Full per-class F1 bar chart is generated at `experiments/results/error_analysis_per_class.png`. Classes falling below F1 = 0.70 are highlighted in red.

---

## Known Limitations and Caveats

1. **Geographic scope:** Trained and evaluated exclusively on German traffic signs. Performance on non-GTSRB sign systems is unknown and likely poor.

2. **Class imbalance:** Test-set imbalance ratio is 12.5:1 (750 vs. 60 images). Macro-F1 weights all classes equally — overall accuracy is biased toward frequent classes. Per-class and per-slice metrics must be reviewed for any safety-relevant application.

3. **Image resolution:** All inputs are downsampled to 32×32 pixels. Fine visual distinctions between similar signs (e.g., 30 km/h vs. 80 km/h at low quality) may be lost.

4. **CNN vs. SVM gap:** The SVM baseline currently outperforms the CNN at this prototype stage. This is expected for small-input, moderate-dataset settings; further CNN training (more epochs, regularization tuning) is expected to close the gap.

5. **Synthetic NLP data:** The TextCNN was trained entirely on synthetically augmented text, not real user queries or sign descriptions. Its generalization to natural human language about signs is not validated.

6. **Calibration:** No probability calibration (e.g., temperature scaling) has been applied. Softmax confidence scores may be overconfident and should not be used as reliability estimates without calibration.

7. **Explainability:** Grad-CAM visualizations are provided for qualitative inspection only. They are not a formal safety or correctness guarantee.

---

## Deployment Guidance

### Prerequisites Before Any Deployment
- [ ] Validate on target-domain sign images (correct country/region)
- [ ] Apply class-weighted training or oversampling to address imbalance
- [ ] Perform probability calibration (e.g., temperature scaling on a held-out val set)
- [ ] Conduct adversarial robustness testing (weather, occlusion, damage)
- [ ] Pass applicable jurisdiction-specific safety certification

### Recommended Integration Pattern (If Extending This Work)
- Treat all predictions with confidence < 0.85 as uncertain; route to human review
- Log all predictions and periodically audit for class-level drift
- Never rely solely on this model for stop/yield/no-entry decisions in any automated context

### Hardware Requirements (Training)
- GPU strongly recommended (CUDA); CPU fallback supported
- Minimum ~4 GB GPU VRAM for batch_size=64, 32×32 images
- Estimated training time: 30 epochs × ~60s/epoch (single mid-range GPU)

---

## Evaluation and Reproducibility

All metrics reported in this card are reproducible by running:

```bash
bash run.sh
```

Full evaluation outputs are saved to:
- `experiments/logs/` — JSON metric files
- `experiments/results/` — plots, confusion matrices, Grad-CAM galleries

Configuration is fully specified in `configs/config.yaml` with seed=42.

---

## Citation

If building on this work, please cite:

```
Half_Exception (2026). Traffic Sign Classifier: Implementing Aspects of RL, NLP, CNN.
CS-304 Project, Holy Angel University. https://github.com/ZelhaDev/TSC-Model
```

---
