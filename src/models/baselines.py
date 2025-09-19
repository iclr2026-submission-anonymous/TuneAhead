import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

def baseline_proxy_ridge(Xtr, ytr, Xval, yval, Xcal, ycal, Xte, alpha=10.0, shrink=0.2):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(Xtr, ytr)
    y_te_raw = np.clip(pipe.predict(Xte), 0.0, 1.0)
    mu = float(np.mean(ytr))
    y_te = (1.0 - shrink)*y_te_raw + shrink*mu
    return y_te

def baseline_reference_ppl(xtr_ppl, ytr, xte_ppl):
    reg = LinearRegression().fit(np.log1p(xtr_ppl.reshape(-1,1)), ytr)
    return reg.predict(np.log1p(xte_ppl.reshape(-1,1)))

def baseline_probe_linear(Xtr, ytr, Xte):
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)

def baseline_early_stop(Xtr, ytr, Xte):
    cols = [c for c in ["initial_loss","loss_decay_rate"] if c in Xtr.columns]
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(Xtr[cols], ytr)
    return pipe.predict(Xte[cols])
