# src/train.py
import argparse
import yaml
from pathlib import Path
import pandas as pd

from src.utils import set_seed
from src.io_utils import read_and_merge
from src.features.build_views import make_feature_views
from src.features.static_rules import select_feature_groups
from src.splitters import split_random, split_group
from src.models.lightgbm_reg import train_lgbm
from src.models.baselines import baseline_proxy_ridge
from src.metrics import all_metrics
from src.conformal import absolute_residual_q, predict_with_ci
from src.models.calibrate import IsotonicCalibrator
from src.plots import pred_vs_true, plot_reliability, grid_pred_vs_true
from src.interpret.shap_global import shap_summary_lightgbm, shap_waterfall_case


def main(cfg_path: str):
    """
    End-to-end pipeline:
      1) Load & merge meta and proxy data
      2) Build feature views (X_all, X_proxy, y)
      3) Train/val/cal/test split
      4) Train LightGBM (Full) + Isotonic calibration
      5) (Optional) Train ablations (Static-only / Dynamic-only) & Proxy baseline
      6) Save small table + figures (pred-vs-true, calibration, grid, SHAP)
    """
    # ========= Load config & set seed =========
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["seed"])

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # ========= 1) Data =========
    df = read_and_merge(
        cfg["data"]["meta_csv"],
        cfg["data"]["proxy_csv"],
        cfg["data"]["key_cols"],
    )

    # Build feature matrices
    X_all, X_proxy, y = make_feature_views(df, cfg["target"])
    static_cols, dynamic_cols = select_feature_groups(X_all)

    # Optional grouping (e.g., dataset_name) for group split
    groups = df["dataset_name"] if "dataset_name" in df.columns else None

    # ========= 2) Split =========
    if cfg["split"]["use_group"] and groups is not None:
        X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_te, y_te = split_group(
            X_all, y, groups
        )
    else:
        X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_te, y_te = split_random(
            X_all,
            y,
            cfg["split"]["test_size"],
            cfg["split"]["val_ratio"],
            cfg["split"]["cal_ratio"],
            cfg["seed"],
        )

    # ========= 3) Train main model (TuneAhead Full) =========
    model_full = train_lgbm(
        X_tr,
        y_tr,
        X_val,
        y_val,
        cfg["seed"],
        cfg["model"]["lgbm"],
    )
    yhat_te_full = model_full.predict(X_te)

    # Conformal interval (optional demonstration)
    q = absolute_residual_q(
        model_full,
        X_cal,
        y_cal,
        cfg["conformal"]["target_coverage"],
    )
    # Example of conformal prediction (not saved, just showing how to use)
    _yhat, _L, _U = predict_with_ci(model_full, X_te, q)

    # Isotonic calibration (fit on validation, apply to test)
    iso = IsotonicCalibrator(y_min=0.0, y_max=1.0).fit(
        model_full.predict(X_val), y_val
    )
    yhat_te_full_cal = iso.predict(yhat_te_full)

    # Metrics
    res_full = all_metrics(y_te, yhat_te_full)
    res_full_cal = all_metrics(y_te, yhat_te_full_cal)

    # ========= 4) Ablations (Static-only / Dynamic-only) =========
    # Static-only
    model_stat = train_lgbm(
        X_tr[static_cols],
        y_tr,
        X_val[static_cols],
        y_val,
        cfg["seed"],
        cfg["model"]["lgbm"],
    )
    yhat_te_stat = model_stat.predict(X_te[static_cols])

    # Dynamic-only
    model_dyn = train_lgbm(
        X_tr[dynamic_cols],
        y_tr,
        X_val[dynamic_cols],
        y_val,
        cfg["seed"],
        cfg["model"]["lgbm"],
    )
    yhat_te_dyn = model_dyn.predict(X_te[dynamic_cols])

    # ========= 5) ProxyLM baseline (if available) =========
    pcols = [c for c in X_proxy.columns if c.endswith("_0shot")]
    rows = [
        {"Model": "TuneAhead (Full)", **res_full},
        {"Model": "TuneAhead (Full, Isotonic)", **res_full_cal},
    ]
    models_for_grid = {
        "TuneAhead (Full)": (y_te, yhat_te_full),
        "Full (Isotonic)": (y_te, yhat_te_full_cal),
        "Static-only": (y_te, yhat_te_stat),
        "Dynamic-only": (y_te, yhat_te_dyn),
    }
    if pcols:
        y_plm = baseline_proxy_ridge(
            X_proxy.loc[X_tr.index, pcols],
            y_tr,
            X_proxy.loc[X_val.index, pcols],
            y_val,
            X_proxy.loc[X_cal.index, pcols],
            y_cal,
            X_proxy.loc[X_te.index, pcols],
        )
        res_proxy = all_metrics(y_te, y_plm)
        rows.append({"Model": "ProxyLM", **res_proxy})
        models_for_grid["ProxyLM"] = (y_te, y_plm)

    # ========= 6) Save table =========
    pd.DataFrame(rows).to_csv(out / "table_main_small.csv", index=False)

    # ========= 7) Save figures =========
    # Pred vs True (Full)
    pred_vs_true(
        y_te,
        yhat_te_full,
        out / "fig_pred_vs_true_full.png",
        title="TuneAhead (Full)",
    )
    # Calibration curve: raw vs isotonic
    plot_reliability(
        y_te,
        yhat_te_full,
        yhat_te_full_cal,
        out / "fig_calibration_full.png",
    )
    # Grid 
    grid_pred_vs_true(
        models_for_grid,
        out / "fig_grid.png",
        title="Pred vs True across methods",
    )

    # SHAP: global summary + one-case waterfall
    shap_summary_lightgbm(
        model_full, X_te, out / "fig_shap_summary.png", max_display=10
    )
    shap_waterfall_case(
        model_full,
        X_te.iloc[[0]],
        out / "fig_shap_waterfall_case0.png",
        title="SHAP Waterfall (test[0])",
    )

    print("Saved outputs to:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    args = ap.parse_args()
    main(args.config)
