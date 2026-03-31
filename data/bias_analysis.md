# Bias Analysis
## Traffic Sign Classifier — Class Imbalance & Subgroup Performance

**Project:** Half_Exception Traffic Sign Classifier (CS-304, SY 2025–2026)  
**Dataset:** GTSRB — `GTfinal_test.csv` (12,630 test images, 43 classes)

---

## 1. Executive Summary

The GTSRB dataset exhibits **significant class imbalance** with a 12.5:1 ratio between the most and least represented classes in the test set. Speed limit signs account for 34.2% of all test images while 13 of 43 classes have 100 or fewer test samples each. This imbalance directly affects model behavior: training loss and accuracy metrics are dominated by high-frequency classes, and per-class performance on rare-but-safety-critical signs is both lower and statistically noisier. This document quantifies the imbalance, analyzes it across sign subgroups, and recommends mitigations.

---

## 2. Overall Imbalance Statistics (Test Set)

| Metric | Value |
|---|---|
| Total test images | 12,630 |
| Number of classes | 43 |
| Most represented class | Class 2 (50 km/h) — 750 images (5.9%) |
| Least represented classes | Classes 0, 19, 27, 32, 37, 41 — 60 images each (0.5%) |
| **Imbalance ratio (max / min)** | **12.5 : 1** |
| Mean images per class | 293.7 |
| Median images per class | ~210 |
| Classes with ≤ 100 test images | **13 of 43 (30.2%)** |
| Classes with ≥ 500 test images | 8 of 43 (18.6%) |

The training set (39,209 images) mirrors this imbalance, as GTSRB's natural capture frequency reflects real-world sign deployment density in Germany — speed limit signs are simply more common on the road than, e.g., "end of all limits" signs.

---

## 3. Subgroup Analysis by Sign Category

The five semantic categories defined in `src/error_analysis.py` are analyzed below. Counts are from the test set.

### 3.1 Speed Limits (Classes 0–8)

| Property | Value |
|---|---|
| Test images | 4,320 (34.2% of total) |
| Classes | 9 |
| Image range per class | 60 (Class 0) – 750 (Class 2) |
| Internal imbalance ratio | 12.5 : 1 |

**Bias concern:** This category dominates the dataset. Overall accuracy is disproportionately influenced by speed limit sign performance. Critically, Class 0 (20 km/h) has only 60 test images — a fraction of what Classes 1 and 2 receive — yet all classes carry equal weight in macro-F1. Speed limit signs are visually similar (same circular red-border design, different numerals), creating a known confusability cluster for CNN models.

### 3.2 Prohibitory Signs (Classes 9, 10, 15, 16, 17)

| Property | Value |
|---|---|
| Test images | 1,860 (14.7%) |
| Classes | 5 |
| Image range per class | 150 – 660 |
| Internal imbalance ratio | 4.4 : 1 |

**Bias concern:** Moderate representation overall. Class 17 (No entry) is safety-critical with 360 test samples — reasonable but not abundant. Classes 15 and 16 (No vehicles, No >3.5t) are at 150 images each. These signs carry severe consequences if missed; their relatively low test-set representation means F1 estimates are less reliable than for speed limit signs.

### 3.3 Warning Signs (Classes 18–31)

| Property | Value |
|---|---|
| Test images | 2,370 (18.8%) |
| Classes | 14 |
| Image range per class | 60 – 480 |
| Internal imbalance ratio | 8.0 : 1 |

**Bias concern:** This is the most internally imbalanced category. It contains 7 of the 13 classes with ≤100 test images:

| Class | Sign | Test Images |
|---|---|---|
| 19 | Dangerous curve left | 60 |
| 20 | Dangerous curve right | 90 |
| 21 | Double curve | 90 |
| 24 | Narrows right | 90 |
| 27 | **Pedestrians** | **60** |
| 28 | Children crossing | 150 |
| 29 | Bicycles crossing | 90 |

Classes 27 (Pedestrians) and 28 (Children crossing) are particularly high-stakes: a miss could mean failure to yield to vulnerable road users. Their low test-set counts (60 and 150 respectively) mean F1 estimates carry high variance; the true deployment error rate could be materially worse than reported metrics suggest.

### 3.4 Mandatory Signs (Classes 33–40)

| Property | Value |
|---|---|
| Test images | 1,770 (14.0%) |
| Classes | 8 |
| Image range per class | 60 – 690 |
| Internal imbalance ratio | 11.5 : 1 |

**Bias concern:** Class 37 (Go straight or left) has only 60 test images against Class 38 (Keep right) with 690. Mandatory direction signs share similar blue-circle designs, creating a confusability risk. Class 37's extreme rarity makes its per-class F1 statistically unreliable.

### 3.5 Informational Signs (Classes 11, 12, 13, 14, 32, 41, 42)

| Property | Value |
|---|---|
| Test images | 2,310 (18.3%) |
| Classes | 7 |
| Image range per class | 60 – 720 |
| Internal imbalance ratio | 12.0 : 1 |

**Bias concern:** This category contains both the highest-stakes signs in the dataset and some of the rarest. Class 14 (Stop) has 270 test images and Class 13 (Yield) has 720 — reasonable representation for critical signs. However, Class 32 (End of all limits) and Class 41 (End of no passing) have only 60 test images each, while Class 42 has 90.

---

## 4. High-Risk Class Inventory

The following classes combine low test-set representation with high real-world safety consequence. These are the highest-priority targets for bias mitigation in future work.

| Class | Sign | Test Images | Safety Concern | Priority |
|---|---|---|---|---|
| 14 | Stop | 270 | Critical — must halt | High |
| 17 | No entry | 360 | Critical — wrong-way entry | High |
| 27 | Pedestrians | 60 | Critical — vulnerable users | **Critical** |
| 28 | Children crossing | 150 | Critical — vulnerable users | **Critical** |
| 19 | Dangerous curve left | 60 | High — collision risk | High |
| 20 | Dangerous curve right | 90 | High — collision risk | High |
| 3 | No passing >3.5t | 450 | Medium | Medium |
| 0 | Speed limit 20 km/h | 60 | Medium — school zones | High |

---

## 5. Impact on Model Metrics

### 5.1 Accuracy vs. Macro-F1 Gap

**Accuracy** (weighted by class frequency) is dominated by Speed Limits and Informational signs — the frequent classes. It will look good even if rare, safety-critical signs perform poorly.

**Macro-F1** (equal weight per class) is a better fairness metric here, because it exposes underperformance on rare classes. The reported gap between the SVM (macro-F1: 90.06%) and CNN (macro-F1: 74.23%) likely reflects substantially worse per-class performance on the 13 minority classes.

**Per-class F1** (generated by `src/error_analysis.py`) is the ground truth for bias auditing. Any class falling below F1 = 0.70 is flagged in `experiments/results/error_analysis_per_class.png`.

### 5.2 Confusion Clustering

Visually similar sign groups create predictable confusion clusters:
- **Speed limit confusions:** 20/30, 80/30, 100/80 (numeral similarity at 32×32 resolution)
- **Warning sign confusions:** Left curve / Right curve / Double curve (similar triangular shape)
- **Mandatory direction confusions:** Turn left / Turn right / Ahead or left / Ahead or right (arrow direction at low resolution)

Top confusion pairs are documented in `experiments/results/error_analysis_top_confusions.png` and `experiments/logs/error_analysis.json`.

### 5.3 Effective Sample Size Problem

With only 60 test images per class for the rarest categories, a model achieving F1 = 0.80 on that class could have a 95% confidence interval spanning ±0.08 or more. Performance claims on minority classes should always be accompanied by sample-size caveats.

---

## 6. Current Mitigations (Implemented)

| Mitigation | Status | Location |
|---|---|---|
| Stratified train/val split | ✅ Implemented | `src/data_pipeline.py` |
| Macro-F1 as primary metric | ✅ Implemented | `src/eval.py`, `src/train.py` |
| Per-class F1 reporting | ✅ Implemented | `src/error_analysis.py` |
| Sign-category slice analysis | ✅ Implemented | `src/error_analysis.py` |
| Top confusion pairs | ✅ Implemented | `src/error_analysis.py` |
| Data augmentation (training) | ✅ Implemented | `src/data_pipeline.py` |
| Ablation: augmentation on vs off | ✅ Implemented | `src/ablation_runner.py` |

---

## 7. Recommended Mitigations (Future Work)

### 7.1 Class-Weighted Loss (High Priority)
Apply inverse-frequency class weights to the cross-entropy loss function to penalize misclassifications on rare classes more heavily:

```python
# In src/train.py — compute weights from training label distribution
from collections import Counter
import torch

label_counts = Counter(all_train_labels)
total = sum(label_counts.values())
weights = torch.tensor([total / (43 * label_counts[c]) for c in range(43)], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### 7.2 Oversampling Minority Classes
Use `torch.utils.data.WeightedRandomSampler` to oversample the 13 minority classes during training, targeting an approximately uniform class distribution per batch.

### 7.3 Targeted Augmentation
Apply heavier augmentation (additional rotations, brightness extremes, blur) specifically to minority classes to artificially expand their effective training set size.

### 7.4 Confidence Calibration
Apply Platt scaling or temperature scaling using the validation set to produce well-calibrated confidence scores. This is especially important for safety-critical rare classes where overconfident wrong predictions are most dangerous.

### 7.5 Threshold Tuning per Class
For deployment in any controlled setting, tune per-class confidence thresholds using the validation set to optimize recall on safety-critical signs (14 — Stop, 17 — No entry, 27 — Pedestrians, 28 — Children crossing) at an acceptable precision cost.

### 7.6 Collect Additional Data for Minority Classes
If this project is extended beyond GTSRB, actively seek additional samples of the 13 minority classes from supplementary datasets or controlled collection to bring all classes to a minimum of 500 training images.

---

## 8. Summary Table: Subgroup Risk Assessment

| Category | % of Test Set | Internal Imbalance | Safety-Critical Classes | Bias Risk Level |
|---|---|---|---|---|
| Speed Limits | 34.2% | 12.5:1 | 0 (20 km/h, 60 imgs) | **Medium** |
| Prohibitory | 14.7% | 4.4:1 | 17 (No entry) | **Medium-High** |
| Warning | 18.8% | 8.0:1 | 27, 28 (Peds, Children) | **High** |
| Mandatory | 14.0% | 11.5:1 | — | **Medium** |
| Informational | 18.3% | 12.0:1 | 14 (Stop) | **High** |

---

## 9. Monitoring Recommendations

Once any version of this classifier is deployed for testing:

1. Log per-class prediction counts and confidence scores during inference.
2. Alert when any class accumulates fewer than 50 predictions per evaluation window — rare classes in inference data signal distribution shift.
3. Re-evaluate per-class F1 quarterly against newly collected ground-truth labels.
4. Track confusion pair frequency over time; a new dominant confusion pair signals either distribution shift or a deteriorating sign condition.

---
