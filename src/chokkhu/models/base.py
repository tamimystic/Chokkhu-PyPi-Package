from __future__ import annotations

import numpy as np


class ChokkhuModel:
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ChokkhuModel:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
