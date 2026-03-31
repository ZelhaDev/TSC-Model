# Data Folder — Traffic Sign Classification Project

This folder stores the GTSRB dataset downloaded by torchvision.  
**Do not commit anything inside `data/` to the repository** — it is listed in `.gitignore`.

---

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)**.

| Property | Detail |
|---|---|
| **Classes** | 43 traffic sign categories |
| **Training images** | 39,209 |
| **Test images** | 12,630 |
| **Source** | Institut für Neuroinformatik, Ruhr-Universität Bochum |
| **License** | Free for academic / non-commercial research. Attribution required (see below). |

No Kaggle account, API key, or manual download is needed.  
`torchvision` handles the download automatically.

---

## Downloading the Data

### Option A — Standalone download script (recommended before first run)

```bash
python data/get_data.py
```

This downloads both the train and test splits to `data/gtsrb/` and confirms image counts.  
If the data already exists, the download is skipped.

### Option B — Automatic download via the pipeline

The full pipeline (`bash run.sh`) triggers the same download automatically at step 1.  
You do not need to run `get_data.py` separately if you are running the full pipeline.

---

## Directory Layout After Download

```
data/
├── get_data.py          ← download script
├── README.md            ← this file
└── gtsrb/               ← created automatically by torchvision
    ├── GTSRB/
    │   ├── Training/    ← 43 class subdirectories, PPM images + CSV annotations
    │   │   ├── 00000/
    │   │   ├── 00001/
    │   │   └── ... (00000–00042)
    │   └── Final_Test/
    │       ├── Images/  ← 12,630 PPM images (00000.ppm – 12629.ppm)
    │       └── GT-final_test.csv
```

> **Note:** The `data/gtsrb/` subdirectory and its contents are created by  
> `torchvision.datasets.GTSRB`. Do not rename or restructure this folder;  
> the data pipeline reads it via torchvision's built-in loader.

---

## Attribution

If you publish results based on GTSRB, cite the original benchmark paper:

> J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel.
> *The German Traffic Sign Recognition Benchmark: A multi-class classification competition.*
> Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN),
> pp. 1453–1460, 2011.

---

## Data Handling Rules

1. **Never commit image data.** `data/` is in `.gitignore`.
2. **Never commit API keys or credentials** of any kind.
3. **Do not modify the test split.** `data/gtsrb/GTSRB/Final_Test/` must remain unaltered to ensure valid, reproducible evaluation.
4. **Use the pipeline.** Always access data through `src/data_pipeline.py` to get consistent preprocessing, normalization, and stratified splits.
5. **Reproduce with the same seed.** All train/val splits use `seed: 42` from `configs/config.yaml`.
