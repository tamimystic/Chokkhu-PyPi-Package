from __future__ import annotations

from typing import Any

import numpy as np


class LinearDiscriminantAnalysis:

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self.scalings_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.means_: dict[Any, np.ndarray] | None = None

    def fit(self, X: Any, y: Any) -> LinearDiscriminantAnalysis:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        n_samples, n_features = X_arr.shape
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("LDA requires at least 2 distinct classes.")

        overall_mean = np.mean(X_arr, axis=0)
        S_W = np.zeros((n_features, n_features), dtype=np.float64)
        S_B = np.zeros((n_features, n_features), dtype=np.float64)
        self.means_ = {}

        for c in self.classes_:
            X_c = X_arr[y_arr == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            self.means_[c] = mean_c
            diff_w = X_c - mean_c
            S_W += diff_w.T @ diff_w
            diff_b = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * (diff_b @ diff_b.T)

        S_W += np.eye(n_features) * 1e-6

        try:
            A = np.linalg.inv(S_W) @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(A)
        except np.linalg.LinAlgError:
            A = np.linalg.pinv(S_W) @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(A)

        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvectors = eigenvectors[:, idx].real

        max_components = min(n_features, n_classes - 1)
        if self.n_components is not None:
            k = min(self.n_components, max_components)
        else:
            k = max_components

        k = max(k, 1)
        self.scalings_ = eigenvectors[:, :k]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.scalings_ is None:
            raise ValueError("LinearDiscriminantAnalysis instance is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.scalings_

    def fit_transform(self, X: Any, y: Any) -> np.ndarray:
        return self.fit(X, y).transform(X)
