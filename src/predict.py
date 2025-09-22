# src/predict.py
# -*- coding: utf-8 -*-
"""
Predict fine-tuning performance for new datasets using a pretrained LightGBM model (.txt).

"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


COMMON_LABEL_NAMES = {
    "y", "label", "target", "acc", "accuracy", "overall_acc", "overall_accuracy",
    "R", "R_ij", "r_ij", "mmlu", "mmlu_acc", "mmlu_accuracy"
}
COMMON_ID_LIKE = {"dataset", "dataset_name", "run_id", "config_id", "pair_id"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict with pretrained LightGBM txt model.")
    ap.add_argument("--model-txt", type=str, default="models/model_TuneAhead_Full.txt",
                    help="Path to a LightGBM Booster .txt model file.")
    ap.add_argument("--data-csv", type=str, required=True,
                    help="Path to CSV containing features for prediction.")
    ap.add_argument("--out-csv", type=str, default="",
                    help="Optional path to save predictions as CSV.")
    ap.add_argument("--id-cols", type=str, nargs="*", default=[],
                    help="Optional columns to carry through to the output (e.g., dataset_name run_id).")
    ap.add_argument("--drop-cols", type=str, nargs="*", default=[],
                    help="Columns to drop before feeding to the model.")
    ap.add_argument("--label-col", type=str, default="",
                    help="If your CSV包含真实标签，可指定列名；将被排除出特征，并一并输出（便于对比）。")
    ap.add_argument("--preview", type=int, default=10,
                    help="Print preview rows of predictions in stdout (default: 10).")
    return ap.parse_args()


def load_model(model_txt: str) -> lgb.Booster:
    if not os.path.exists(model_txt):
        raise FileNotFoundError(f"Model file not found: {model_txt}")
    booster = lgb.Booster(model_file=model_txt)
    return booster


def choose_feature_columns(df: pd.DataFrame,
                           booster: lgb.Booster,
                           drop_cols: List[str],
                           label_col: str) -> List[str]:
    """
    """

    to_drop = set(drop_cols)
    if label_col:
        to_drop.add(label_col)


    feat_names = booster.feature_name()
    feat_names = [f for f in feat_names if f is not None] if feat_names else []


    if feat_names:
        inter = [c for c in feat_names if c in df.columns and c not in to_drop]
        if len(inter) > 0:
            return inter


    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    exclude = COMMON_LABEL_NAMES | COMMON_ID_LIKE | to_drop
    fallback = [c for c in numeric_cols if c not in exclude]

    if len(fallback) == 0:

        convertible = []
        for c in df.columns:
            if c in exclude:
                continue
            try:
                pd.to_numeric(df[c], errors="raise")
                convertible.append(c)
            except Exception:
                pass
        fallback = convertible

    if len(fallback) == 0:
        raise ValueError("No usable feature columns found. "
                         "Check your CSV columns and --drop-cols/--label-col settings.")
    return fallback


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """

    """
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # 用列中位数填充 NaN
    na_counts = X.isna().sum().sum()
    if na_counts > 0:
        med = X.median(numeric_only=True)
        X = X.fillna(med)
    return X


def main():
    args = parse_args()

  
    booster = load_model(args.model_txt)
    df = pd.read_csv(args.data_csv)

    
    carry_cols = []
    for c in (args.id_cols or []):
        if c in df.columns:
            carry_cols.append(c)
    if args.label_col and args.label_col in df.columns:
        carry_cols.append(args.label_col)

    
    feature_cols = choose_feature_columns(df, booster, args.drop_cols, args.label_col)
    X = coerce_numeric(df, feature_cols)

    
    preds = booster.predict(X)  # LightGBM Booster 直接接受 numpy/pandas

    
    out_df = pd.DataFrame({"prediction": preds})
    
    if carry_cols:
        out_df = pd.concat([df[carry_cols].reset_index(drop=True), out_df], axis=1)

    
    n_prev = max(1, int(args.preview))
    print("\n[Predict] feature columns used ({}):".format(len(feature_cols)))
    print(feature_cols[:20], "..." if len(feature_cols) > 20 else "")
    print("\n[Predict] preview (first {} rows):".format(n_prev))
    print(out_df.head(n_prev).to_string(index=False))

  
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"\n[Predict] Saved predictions to: {args.out_csv}")
    else:
        print("\n[Predict] Tip: use --out-csv results/predictions.csv to save predictions.")


if __name__ == "__main__":
    main()
