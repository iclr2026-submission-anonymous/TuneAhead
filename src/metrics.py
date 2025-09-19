import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y, yhat): return float(np.sqrt(mean_squared_error(y, yhat)))
def r2(y, yhat):   return float(r2_score(y, yhat))
def pearson(y, yhat): 
    c = np.corrcoef(y, yhat)
    return float(c[0,1]) if np.isfinite(c[0,1]) else 0.0

def acc_within_pp(y, yhat, k_pp):
    return float(np.mean(np.abs(y - yhat) <= k_pp/100.0))

def all_metrics(y, yhat):
    return {
        "RMSE": rmse(y,yhat),
        "R2": r2(y,yhat),
        "Pearson": pearson(y,yhat),
        "Acc@1pp": acc_within_pp(y,yhat,1.0),
        "Acc@2pp": acc_within_pp(y,yhat,2.0),
        "Acc@3pp": acc_within_pp(y,yhat,3.0),
    }
