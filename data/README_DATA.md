# 📊 Data for TuneAhead

This folder contains the released **subset(s)** of the meta-dataset used in our experiments.  
All files are stored in **CSV format** for ease of use.

## 📂 Files

- `sample_metadataset.csv`  
  A randomly sampled subset of the full meta-dataset used in the paper.  
  Each row corresponds to a fine-tuning run, with columns describing:
  - **Static features** (dataset descriptors, e.g., lexical diversity, duplication ratio)
  - **Dynamic features** (100-step probe signals, e.g., loss decay, gradient stability)
  - **Ground-truth performance** (MMLU accuracy of the full fine-tune)

- `sample_proxy.csv`  
  A placeholder subset showing the expected format for ProxyLM-based predictors.  
  The full ProxyLM meta-dataset is **not released** due to reliance on proprietary pretrained models.

## 🔍 Notes

- **Format**: All subsets are in CSV format. Each row = one run, each column = one feature or label.  
- **Full-scale datasets** are **not included** due to license/privacy constraints and ongoing research.  
- The released subset mirrors the same schema and interface as the full dataset, so users can directly swap in their own data.  
- Feature definitions and extraction methods are documented in `src/features/` and Appendix B of the paper.

## ▶️ Usage

By default, `train.py` expects the data path to point to a CSV in this folder:

```bash
python src/train.py --data data/sample_metadataset.csv --config configs/main.yaml
