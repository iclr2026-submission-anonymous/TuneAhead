from pathlib import Path
import pandas as pd

def read_and_merge(meta_csv, proxy_csv, key_cols):
    df_meta = pd.read_csv(meta_csv)
    df_proxy = pd.read_csv(proxy_csv)
    assert all(k in df_meta.columns for k in key_cols)
    assert all(k in df_proxy.columns for k in key_cols)
    df_proxy = df_proxy.drop_duplicates(subset=key_cols, keep="last")
    df = df_meta.merge(df_proxy, on=key_cols, how="left")

    # normalize proxy columns to [0,1]
    prox_cols = [c for c in df.columns if c.startswith("proxylm_")]
    for c in prox_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() is not None and df[c].max() > 1.5:
            df[c] = df[c] / 100.0
        df[c] = df[c].fillna(df[c].median())
    return df
