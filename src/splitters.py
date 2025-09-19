from sklearn.model_selection import train_test_split, GroupKFold

def split_random(X, y, test_size, val_ratio, cal_ratio, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_tr2, X_tmp, y_tr2, y_tmp = train_test_split(X_tr, y_tr, test_size=(val_ratio+cal_ratio), random_state=seed)
    X_val, X_cal, y_val, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
    return X_tr2, y_tr2, X_val, y_val, X_cal, y_cal, X_te, y_te

def split_group(X, y, groups, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        X_tr_raw, y_tr_raw = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_tr_raw, y_tr_raw, test_size=0.4, random_state=seed)
        X_val, X_cal, y_val, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)
        return X_tr, y_tr, X_val, y_val, X_cal, y_cal, X_te, y_te
