import lightgbm as lgb

def train_lgbm(Xtr, ytr, Xval, yval, seed, params):
    model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, **params)
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model
