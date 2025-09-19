import numpy as np

def make_feature_views(df, target, drop_id_cols=("dataset_name",)):
    y = df[target].astype(float)
    drop_cols = [c for c in (target,)+drop_id_cols if c in df.columns]
    Z = (df.drop(columns=drop_cols, errors="ignore")
           .select_dtypes(include=[np.number])
           .replace([np.inf,-np.inf], np.nan))
    Z = Z.fillna(Z.median())

    # 主视图：不含 proxylm_*
    X_all = Z.drop(columns=[c for c in Z.columns if c.startswith("proxylm_")], errors="ignore")

    # Proxy 视图
    proxy_cols = [c for c in Z.columns if c.startswith("proxylm_")]
    extra = [c for c in ["jsd_train_test","tfidf_cosine","sbert_cosine","ttr","vocab_size","avg_input_length"]
             if c in Z.columns]
    X_proxy = Z[proxy_cols].copy()
    for c in extra: X_proxy[c] = Z[c]
    return X_all, X_proxy, y
