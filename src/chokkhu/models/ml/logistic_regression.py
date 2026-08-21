from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class LogisticRegression(ChokkhuModel):
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        regularization: str | None = None,
        alpha: float = 1.0,
        tolerance: float = 1e-4,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.tolerance = tolerance
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LogisticRegression:
        if y is None:
            raise ValueError("y cannot be None for Logistic Regression")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        try:
            from tqdm import tqdm

            iterator = tqdm(range(self.epochs), desc="Training Logistic Regression")
        except ImportError:
            iterator = range(self.epochs)

        for _ in iterator:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            if self.regularization == "l2":
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == "l1":
                dw += (self.alpha / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * float(db)

            if np.max(np.abs(self.learning_rate * dw)) < self.tolerance:
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return np.where(probas >= 0.5, 1, 0)
