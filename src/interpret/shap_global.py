import shap, matplotlib.pyplot as plt

def shap_summary_lightgbm(lgbm_model, X, out_png, max_display=10, title="SHAP Summary"):
    explainer = shap.TreeExplainer(lgbm_model)
    sv = explainer.shap_values(X)
    fig = plt.figure()
    shap.summary_plot(sv, X, max_display=max_display, show=False)
    plt.title(title, pad=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def shap_waterfall_case(lgbm_model, X_row, out_png, title="SHAP Waterfall (one case)"):
    explainer = shap.TreeExplainer(lgbm_model)
    sv = explainer.shap_values(X_row)
    base = explainer.expected_value
    shap.plots._waterfall.waterfall_legacy(base, sv[0], feature_names=X_row.columns, max_display=12, show=False)
    plt.title(title, pad=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
