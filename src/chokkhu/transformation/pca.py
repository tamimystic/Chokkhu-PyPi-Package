from __future__ import annotations

from typing import Any

import numpy as np


class PCA:

    def __init__(
        self,
        n_components: int | None = None,
        variance_ratio: float | None = None,
        whiten: bool = False,
    ) -> None:
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.whiten = whiten
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: Any) -> PCA:
        arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = arr.shape
        self.mean_ = np.mean(arr, axis=0)
        X_centered = arr - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        if cov_matrix.ndim == 0:
            cov_matrix = cov_matrix.reshape((1, 1))

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_variance = float(np.sum(eigenvalues))
        if total_variance > 0:
            ratio = eigenvalues / total_variance
        else:
            ratio = np.zeros_like(eigenvalues)

        if self.variance_ratio is not None and 0.0 < self.variance_ratio <= 1.0:
            cum_var: np.ndarray = np.cumsum(ratio)
            k = int(np.searchsorted(cum_var, self.variance_ratio) + 1)
            k = min(k, n_features)
        elif self.n_components is not None:
            k = min(self.n_components, n_features)
        else:
            k = n_features

        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.components_ = eigenvectors[:, :k]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA instance is not fitted yet.")
        arr = np.asarray(X, dtype=np.float64)
        X_centered = arr - self.mean_
        X_transformed = X_centered @ self.components_
        if self.whiten and self.explained_variance_ is not None:
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-10))
            X_transformed /= scale
        return X_transformed

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: Any) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA instance is not fitted yet.")
        arr = np.asarray(X_transformed, dtype=np.float64)
        if self.whiten and self.explained_variance_ is not None:
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-10))
            arr = arr * scale
        return (arr @ self.components_.T) + self.mean_


class TruncatedSVD:

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, X: Any) -> TruncatedSVD:
        arr = np.asarray(X, dtype=np.float64)
        _, S, Vt = np.linalg.svd(arr, full_matrices=False)
        k = min(self.n_components, arr.shape[1])
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.components_ is None:
            raise ValueError("TruncatedSVD instance is not fitted yet.")
        arr = np.asarray(X, dtype=np.float64)
        return arr @ self.components_.T

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)
