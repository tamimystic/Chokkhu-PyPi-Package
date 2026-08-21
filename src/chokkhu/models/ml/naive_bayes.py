from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class NaiveBayes(ChokkhuModel):
    def __init__(self) -> None:
        self.classes: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.priors: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NaiveBayes:
        if y is None:
            raise ValueError("y cannot be None for Naive Bayes")
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples)

        return self

    def _pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.var is None:
            raise ValueError("Model is not fitted yet.")
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_log_proba_single(self, x: np.ndarray) -> np.ndarray:
        if self.classes is None or self.priors is None:
            raise ValueError("Model is not fitted yet.")
        posteriors = []
        for idx in range(len(self.classes)):
            prior = np.log(self.priors[idx])
            posterior: float = float(np.sum(np.log(self._pdf(idx, x) + 1e-9)))
            posteriors.append(prior + posterior)
        return np.array(posteriors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise ValueError("Model is not fitted yet.")
        y_pred = [self.classes[np.argmax(self._predict_log_proba_single(x))] for x in X]
        return np.array(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise ValueError("Model is not fitted yet.")
        probas = []
        for x in X:
            log_posteriors = self._predict_log_proba_single(x)
            # Subtract max for numerical stability before exp
            log_posteriors -= np.max(log_posteriors)
            exp_posteriors = np.exp(log_posteriors)
            probas.append(exp_posteriors / np.sum(exp_posteriors))
        return np.array(probas)
