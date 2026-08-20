import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.nanmean(X_arr, axis=0)
        self.std = np.nanstd(X_arr, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr * self.std + self.mean

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.min = np.nanmin(X_arr, axis=0)
        self.max = np.nanmax(X_arr, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        rng = self.max - self.min
        rng[rng == 0] = 1.0
        norm = (X_arr - self.min) / rng
        return norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        rng = self.max - self.min
        rng[rng == 0] = 1.0
        unscaled = (X_arr - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return unscaled * rng + self.min

class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.median = np.nanmedian(X_arr, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.median) / (self.iqr + 1e-8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr * self.iqr + self.median

class MaxAbsScaler:
    def __init__(self):
        self.max_abs = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.max_abs = np.nanmax(np.abs(X_arr), axis=0)
        self.max_abs[self.max_abs == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr / self.max_abs

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.max_abs

class L2Scaler:
    def fit(self, X):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X_arr / norms

    def fit_transform(self, X):
        return self.transform(X)

def get_scaler(name: str, **kwargs):
    scalers = {
        "standard": StandardScaler,
        "minmax": lambda: MinMaxScaler(feature_range=kwargs.get("feature_range", (0, 1))),
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "l2": L2Scaler
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler: {name}")
    return scalers[name]()
