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
├── preprocessing/              # Get Ground Truth and Feature extraction
│   ├── Feature Extraction Layer
│        ├── feature_pipeline.py       # Unified pipeline
│        ├── static_features.py        # Static features
│        ├── dynamic_probes.py         # Dynamic features
│        └── pipe_folder_to_csv_isolated.py  # Isolated processing
│   ├── batch_mmlu_eval.py      # Batch MMLU evaluation controller
│   ├── auto_mmlu_eval.py      # Automated experiment orchestrator
│   ├── evaluate_model.py      # Model evaluation module
│   └── train_model.py         # LoRA training module
├── run.sh                  # Example script for training & evaluation
├── configs/
│   └── main.yaml           # Default configuration
├── model/
│   ├──  models/Qwen/Qwen2.5-7B-Instruct-1M
│   └──  download_qwen.py
├── data/
│   ├── README_DATA.md
│   ├── mmlu_dataset/test_all_subjects.json
│   ├── train_data/chunk_001/*.json
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
    │   ├── build_views.py  # Build design matrices (X_all/X_proxy/y) from merged CSV; no feature computation
    │   └── static_rules.py # Heuristically partition columns into static/dynamic/hyper groups for ablations
    └── interpret/
        └── shap_global.py  # SHAP-based interpretability
└── README.md
```
### 📥 Download Model (Qwen)

```bash
python model/download_qwen.py
```
## 🧩 Two Subsystems

- Training & Evaluation (LoRA + MMLU)
  - Purpose: LoRA fine-tuning with MMLU evaluation; supports batch and single experiments.
  - Run :
    ```bash
    MODEL_NAME="model/models/Qwen/Qwen2.5-7B-Instruct-1M" DATA_FOLDER="data/train_data/chunk_001" MMLU_PATH="data/mmlu_dataset/test_all_subjects.json" python preprocessing/batch_mmlu_eval.py
    ```


- Feature Extraction (Static + Dynamic)
  - Purpose: Extract static features (text stats, semantics, perplexity) and dynamic features (early fine-tuning signals); output unified CSV.
  - Run :
    ```bash
    mkdir -p results && python -m preprocessing.pipe_folder_to_csv_isolated --data_folder data/train_data/chunk_001 --model_path model/models/Qwen/Qwen2.5-7B-Instruct-1M --output_csv results/features_chunk_001.csv --sample_size 100 --batch_size 1 --probe_steps 20
    # Note: set --sample_size 0 to use the entire dataset.
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

## 🔮 Using Pretrained Model

We provide a pretrained **TuneAhead (Full)** LightGBM model in `models/model_TuneAhead_Full.txt`,  
which can be used to directly predict fine-tuning performance for new datasets.

### 1. Prepare your dataset
- Input format: **CSV** file containing the same static and dynamic features used in our paper.  
- You may optionally include identifier columns (e.g., `dataset_name`, `run_id`) and ground-truth labels (e.g., `overall_accuracy`) for comparison.  
- Example: `data/sample_metadataset.csv`

### 2. Run prediction
```bash
# Minimal usage: only specify your CSV file
python src/predict.py --data-csv data/sample_metadataset.csv

# With additional options
python src/predict.py \
  --model-txt models/model_TuneAhead_Full.txt \
  --data-csv data/sample_metadataset.csv \
  --id-cols dataset_name run_id \
  --label-col overall_accuracy \
  --out-csv results/predictions.csv
```

## 📜 License & Notes

- Full-scale meta-datasets are **not released** due to license/privacy constraints and because they remain part of ongoing research.  
- All released code is anonymized and stripped of sensitive components for ICLR review.
