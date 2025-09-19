import numpy as np

def absolute_residual_q(model, X_cal, y_cal, target_cov=0.86):
    cal_pred = model.predict(X_cal)
    q = np.quantile(np.abs(y_cal - cal_pred), target_cov)
    return float(q)

def predict_with_ci(model, Xte, q):
    yhat = model.predict(Xte)
    return yhat, yhat - q, yhat + q
