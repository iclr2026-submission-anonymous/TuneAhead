import numpy as np, matplotlib.pyplot as plt

def pred_vs_true(y, yhat, path, to_pct=True, title=""):
    yt = y*100 if to_pct else y
    yp = yhat*100 if to_pct else yhat
    fig, ax = plt.subplots(figsize=(4.2,4.0))
    ax.scatter(yt, yp, s=14, alpha=0.75, edgecolors="none")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo,hi],[lo,hi], ls=(0,(4,2)), c="#666", lw=1.0)
    ax.set_xlabel("True Accuracy (%)"); ax.set_ylabel("Predicted Accuracy (%)")
    ax.set_title(title, pad=6); ax.grid(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)

def calibration_curve_reg(y_true, y_pred, n_bins=10):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    bins = np.quantile(y_pred, np.linspace(0,1,n_bins+1))
    bins[0] -= 1e-9; bins[-1] += 1e-9
    idx = np.digitize(y_pred, bins) - 1
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0: 
            continue
        xs.append(y_pred[mask].mean())
        ys.append(y_true[mask].mean())
    return np.array(xs), np.array(ys)

def plot_reliability(y_true, y_pred_raw, y_pred_cal, path, title="Calibration (Regression)"):
    xs_raw, ys_raw = calibration_curve_reg(y_true, y_pred_raw)
    xs_cal, ys_cal = calibration_curve_reg(y_true, y_pred_cal)
    lo = min(xs_raw.min(), ys_raw.min(), xs_cal.min(), ys_cal.min())
    hi = max(xs_raw.max(), ys_raw.max(), xs_cal.max(), ys_cal.max())

    fig, ax = plt.subplots(figsize=(4.2,4.0))
    ax.plot([lo,hi],[lo,hi], ls=(0,(4,2)), lw=1.0)           # ideal
    ax.plot(xs_raw, ys_raw, marker="o", label="Raw")
    ax.plot(xs_cal, ys_cal, marker="s", label="Isotonic")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.set_title(title, pad=6); ax.grid(True); ax.legend()
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)

def grid_pred_vs_true(models_dict, path, to_pct=True, ncols=3, title="Pred vs True across methods"):
    import math
    K = len(models_dict); ncols = min(ncols, K); nrows = math.ceil(K/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.0*nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for ax in axes.ravel(): ax.axis("off")
    for i, (name, (y, yhat)) in enumerate(models_dict.items()):
        r, c = divmod(i, ncols)
        ax = axes[r, c]; ax.axis("on")
        yt = y*100 if to_pct else y
        yp = yhat*100 if to_pct else yhat
        ax.scatter(yt, yp, s=12, alpha=0.75, edgecolors="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo,hi],[lo,hi], ls=(0,(4,2)), c="#666", lw=1.0)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("True (%)"); ax.set_ylabel("Pred (%)")
        ax.grid(True)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
    fig.suptitle(title, y=0.99, fontsize=12)
    plt.tight_layout(rect=[0,0,1,0.98])
    fig.savefig(path, bbox_inches="tight", dpi=300); plt.close(fig)
