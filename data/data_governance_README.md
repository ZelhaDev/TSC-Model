# Dataset Governance
## Traffic Sign Classifier — Data Directory

**Project:** Half_Exception Traffic Sign Classifier (CS-304, SY 2025–2026)  
**Team:** Ahmad, Saeed · Aveena, Stephen · Predilla, Stanley · Siron, Carlo

---

## 1. Dataset Identity

| Property | Detail |
|---|---|
| **Name** | German Traffic Sign Recognition Benchmark (GTSRB) |
| **Version used** | Final release (IJCNN 2011 competition set) |
| **Source** | Institut für Neuroinformatik, Ruhr-Universität Bochum, Germany |
| **Official site** | http://benchmark.ini.rub.de |
| **Contact (original)** | tsr-benchmark@ini.rub.de |
| **Torchvision access** | `torchvision.datasets.GTSRB(root=..., download=True)` |
| **Local path (after download)** | `data/gtsrb/GTSRB/` (created automatically by torchvision) |

---

## 2. License and Terms of Use

GTSRB is published for **free academic and non-commercial research use**. The dataset does not carry a standard OSI-approved open-source license; rather, it is released under the terms of the original competition:

- **Permitted:** Academic research, benchmarking, educational use, publication of results derived from the dataset.
- **Required:** Attribution to the Institut für Neuroinformatik and the IJCNN 2011 benchmark when publishing results.
- **Prohibited:** Commercial redistribution of the raw images, use in products without explicit permission from the original authors.

**This project does not redistribute GTSRB images.** The `data/` directory is listed in `.gitignore`. Images are downloaded to each contributor's local machine and are not committed to the repository.

### Attribution (required in any publication)
> J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. *The German Traffic Sign Recognition Benchmark: A multi-class classification competition.* In Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1453–1460, 2011.

---

## 3. Dataset Structure

### 3.1 Local Directory Layout

After running `python data/get_data.py` (or `bash run.sh`), torchvision creates:

```
data/
├── get_data.py              ← download script
├── README.md
└── gtsrb/                   ← created automatically by torchvision
    └── GTSRB/
        ├── Training/
        │   ├── 00000/       # Class 0 — Speed limit 20 km/h
        │   │   ├── 00000_00000.ppm
        │   │   ├── GT-00000.csv   # per-class annotation file
        │   │   └── ...
        │   ├── 00001/       # Class 1 — Speed limit 30 km/h
        │   └── ... (43 class folders, 00000–00042)
        └── Final_Test/
            ├── Images/
            │   ├── 00000.ppm    # images in random order, no class folders
            │   └── ... (12,630 images total)
            └── GT-final_test.csv
```

> The directory structure inside `data/gtsrb/` is managed entirely by torchvision and must not be manually renamed or restructured. `src/data_pipeline.py` reads it via `torchvision.datasets.GTSRB`.

### 3.2 Image Format

- **Format:** PPM (Portable Pixmap, RGB color)
- **Naming (training):** `XXXXX_YYYYY.ppm` — `XXXXX` = track number (same physical sign), `YYYYY` = frame index within track
- **Naming (test):** Sequential integer (`00000.ppm` → `12629.ppm`), random class order

### 3.3 Annotation Format

CSV files use semicolons (`;`) as delimiters. Fields:

| Field | Type | Description |
|---|---|---|
| `Filename` | string | Image filename |
| `Width` | int | Image width in pixels |
| `Height` | int | Image height in pixels |
| `Roi.X1`, `Roi.Y1` | int | Top-left corner of sign bounding box |
| `Roi.X2`, `Roi.Y2` | int | Bottom-right corner of sign bounding box |
| `ClassId` | int | Ground-truth class label (0–42) |

> The test annotation file used in this project is `GTfinal_test.csv` (semicolon-delimited, 12,630 rows + header).

---

## 4. Dataset Statistics

### 4.1 Split Summary

| Split | Images | Source |
|---|---|---|
| Training (full) | 39,209 | `Final_Training/` |
| Training subset (after 80/20 split) | ~31,367 | Used for model training |
| Validation subset | ~7,842 | Held out from training, stratified |
| Test | 12,630 | `Final_Test/` / `GTfinal_test.csv` |

### 4.2 Class Distribution — Test Set

The test set contains **12,630 images across 43 classes** with a pronounced long-tail distribution:

| Statistic | Value |
|---|---|
| Most frequent class | Class 2 (50 km/h) — 750 images |
| Least frequent classes | Classes 0, 19, 27, 32, 37, 41 — 60 images each |
| Imbalance ratio (max/min) | **12.5 : 1** |
| Mean images per class | 293.7 |
| Median images per class | ~210 |

**Full per-class test distribution:**

| Class | Sign | Count | Class | Sign | Count |
|---|---|---|---|---|---|
| 0 | Speed limit 20 km/h | 60 | 22 | Bumpy road | 120 |
| 1 | Speed limit 30 km/h | 720 | 23 | Slippery road | 150 |
| 2 | Speed limit 50 km/h | 750 | 24 | Narrows right | 90 |
| 3 | Speed limit 60 km/h | 450 | 25 | Road work | 480 |
| 4 | Speed limit 70 km/h | 660 | 26 | Traffic signals | 180 |
| 5 | Speed limit 80 km/h | 630 | 27 | Pedestrians | 60 |
| 6 | End of 80 km/h | 150 | 28 | Children crossing | 150 |
| 7 | Speed limit 100 km/h | 450 | 29 | Bicycles crossing | 90 |
| 8 | Speed limit 120 km/h | 450 | 30 | Ice/snow | 150 |
| 9 | No passing | 480 | 31 | Wild animals | 270 |
| 10 | No pass >3.5t | 660 | 32 | End of all limits | 60 |
| 11 | Right of way | 420 | 33 | Turn right | 210 |
| 12 | Priority road | 690 | 34 | Turn left | 120 |
| 13 | Yield | 720 | 35 | Ahead only | 390 |
| 14 | Stop | 270 | 36 | Ahead or right | 120 |
| 15 | No vehicles | 210 | 37 | Ahead or left | 60 |
| 16 | No >3.5t | 150 | 38 | Keep right | 690 |
| 17 | No entry | 360 | 39 | Keep left | 90 |
| 18 | General caution | 390 | 40 | Roundabout | 90 |
| 19 | Dangerous curve left | 60 | 41 | End no passing | 60 |
| 20 | Dangerous curve right | 90 | 42 | End no pass >3.5t | 90 |
| 21 | Double curve | 90 | | **Total** | **12,630** |

### 4.3 Image Dimensions

| Statistic | Value |
|---|---|
| Minimum size | 25 × 25 pixels |
| Maximum size | 266 × 232 pixels |
| Mean size | ~51 × 50 pixels |
| Working resolution (after pipeline) | 32 × 32 pixels |

---

## 5. Consent and Data Collection

GTSRB images were captured by the Institut für Neuroinformatik on public roads in Germany. As a dataset of public infrastructure (road signs) rather than of individuals, no individual consent was required or obtained for the sign images themselves. However:

- Some images may **incidentally capture license plates or partial pedestrian imagery** due to roadside capture conditions.
- The original collectors are solely responsible for compliance with German data protection law (BDSG, now superseded by GDPR) at the time of collection (pre-2011).
- **This project performs no re-identification, personal data extraction, or privacy-relevant processing** on any image content beyond sign classification.
- If this work is extended to include images captured by the project team (e.g., new field data), a fresh data governance review and applicable privacy impact assessment must be conducted before those images are added to the dataset.

---

## 6. Representativeness — Justification and Limitations

### What the dataset represents well
- German traffic signs under typical Central European roadside conditions
- A wide range of physical sign states: varying lighting, distances, viewing angles, and minor weathering
- The temporal sequence of signs (track structure in training set) captures natural variation

### What the dataset does not represent
| Gap | Consequence |
|---|---|
| **Only German signs** | Models fail on non-Vienna-Convention sign designs without domain adaptation |
| **No severe weather** | Performance under heavy rain, snow, fog, or night-time conditions is unknown |
| **No heavily damaged signs** | Severely vandalized, occluded, or faded signs are underrepresented |
| **No construction variants** | Temporary signs or modified signs are not covered |
| **No sequential video context** | Images are single frames; no temporal smoothing is possible with this data alone |
| **Pre-2011 capture** | Modern sign materials, retroreflective coatings, or LED signs may not be represented |

These gaps directly inform the geographic scope and deployment restrictions stated in the Model Card and Ethics Statement.

---

## 7. Download Instructions

### Via torchvision (automatic — the only method used in this project)

```python
from torchvision import datasets
train_dataset = datasets.GTSRB(root="data", split="train", download=True)
test_dataset  = datasets.GTSRB(root="data", split="test",  download=True)
```

Or use the provided convenience script:

```bash
python data/get_data.py
```

No Kaggle account, API key, or manual download is required. The download is skipped automatically if the data already exists at `data/gtsrb/`.

---

## 8. Data Handling Rules for Contributors

1. **Never commit image data.** The `data/` directory is in `.gitignore`. Keep it that way.
2. **Never commit API keys.** Do not hard-code or commit `kaggle.json` or any credentials.
3. **Use the pipeline.** Always access data through `src/data_pipeline.py` to ensure consistent preprocessing, normalization, and splits.
4. **Reproduce with the same seed.** All splits use `seed: 42` from `configs/config.yaml` for reproducibility.
5. **Do not modify the raw test split.** `data/gtsrb/GTSRB/Final_Test/` must remain unaltered to ensure valid evaluation.

---
