import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    """Monotone calibration for regression (maps raw yhat -> calibrated yhat)."""
    def __init__(self, y_min=0.0, y_max=1.0):
        self.y_min, self.y_max = y_min, y_max
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, yhat_val, y_val):
        self._iso.fit(np.asarray(yhat_val), np.asarray(y_val))
        return self

    def predict(self, yhat):
        ycal = self._iso.predict(np.asarray(yhat))
        return np.clip(ycal, self.y_min, self.y_max)
