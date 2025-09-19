# TuneAhead (Anonymous Repo)

## 📦 Installation

```bash
git clone https://anonymous.4open.science/r/tuneahead-anonymous.git
cd tuneahead-anonymous
pip install -r requirements.txt
```

## 📂 Repository Structure

```
TuneAhead/
├── requirements.txt 
├── features/              # Feature extraction modules
│   ├── extract_static.py  # Static dataset descriptors
│   ├── extract_dynamic.py # Dynamic probe signals
│   └── __init__.py
├── run.sh                  # Example script for training & evaluation
├── configs/
│   └── main.yaml           # Default configuration
├── data/
│   └── README_DATA.md
│   ├── sample_metadataset.csv 
└── src/
    ├── io_utils.py         # Data I/O helpers
    ├── utils.py            # General utilities
    ├── splitters.py        # Dataset splitting
    ├── metrics.py          # Evaluation metrics
    ├── conformal.py        # (Optional) conformal calibration
    ├── plots.py            # Visualization helpers (grid, SHAP, calibration)
    ├── train.py            # Main training script
    ├── models/
    │   ├── lightgbm_reg.py # Gradient boosting predictor
    │   ├── baselines.py    # Baseline models (ProxyLM, scaling-law, etc.)
    │   └── calibrate.py    # Isotonic/Platt calibration
    ├── features/
    │   ├── build_views.py  # Dynamic probe features
    │   └── static_rules.py # Static dataset descriptors
    └── interpret/
        └── shap_global.py  # SHAP-based interpretability
└── README.md
```

## 📊 Data

We release a randomly sampled subset of the meta-dataset under `data/`.  
All subsets are stored in **CSV format** and can be directly consumed by the training and evaluation scripts.

Full-scale datasets and private ETL pipelines are **not included** due to license/privacy constraints and ongoing research.  

All interfaces remain identical, so replacing the subset CSVs with full datasets will work without modification.

## 🛠 Feature Extraction

Code for feature extraction is under `features/`:

- `extract_static.py`: computes dataset-intrinsic descriptors (e.g., lexical diversity, token length statistics, duplication rate).  
- `extract_dynamic.py`: runs fixed-budget probe fine-tunes to capture early dynamics (e.g., loss decay, gradient stability).  

Both modules output CSV files that match the input format expected by `train.py`.

## 🚀 Training & Evaluation
Example usage (see run.sh for more):
```bash
# Train predictor
bash run.sh
```

## 📈 Visualization

Visualization scripts are under `plots/`:

- Grid plots of predicted vs. true accuracy  
- SHAP feature attribution plots  
- Calibration and isotonic regression curves  

Run, for example:

```bash
python plots/plot_grid.py --input results/predictions.csv
```

---

## 📜 License & Notes

- Full-scale meta-datasets are **not released** due to license/privacy constraints and because they remain part of ongoing research.  
- All released code is anonymized and stripped of sensitive components for ICLR review.  
