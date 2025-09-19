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
├── features/              # Feature extraction modules
│   ├── extract_static.py  # Static dataset descriptors
│   ├── extract_dynamic.py # Dynamic probe signals
│   └── __init__.py
├── train.py               # Main training script for predictors
├── evaluate.py            # Evaluation and metric reporting
├── plots/                 # Visualization scripts (grid plots, SHAP, calibration)
├── data/                  # Example subset of the meta-dataset (CSV format)
├── utils/                 # Helper functions (data loading, calibration, conformal)
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

```bash
# Train predictor
python train.py --config configs/default.yaml

# Evaluate predictor
python evaluate.py --checkpoint checkpoints/model.pkl
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
