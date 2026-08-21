from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class LinearRegression(ChokkhuModel):
    def __init__(
        self,
        method: str = "ols",
        regularization: str | None = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-4,
    ):
        self.method = method
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> LinearRegression:
        if y is None:
            raise ValueError("y cannot be None for Linear Regression")
        n_samples, n_features = X.shape

        if self.method == "ols" and self.regularization is None:
            X_b = np.c_[np.ones((n_samples, 1)), X]
            try:
                theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = float(theta[0])
            self.weights = theta[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            try:
                from tqdm import tqdm

                iterator = tqdm(range(self.epochs), desc="Training Linear Regression")
                has_tqdm = True
            except ImportError:
                iterator = range(self.epochs)
                has_tqdm = False

            for epoch in iterator:
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y

                dw = (1 / n_samples) * np.dot(X.T, error)
                db = (1 / n_samples) * np.sum(error)

                if self.regularization == "l2" or self.regularization == "ridge":
                    dw += (self.alpha / n_samples) * self.weights
                elif self.regularization == "l1" or self.regularization == "lasso":
                    dw += (self.alpha / n_samples) * np.sign(self.weights)
                elif self.regularization == "elastic_net":
                    l1_term = self.l1_ratio * np.sign(self.weights)
                    l2_term = (1 - self.l1_ratio) * self.weights
                    dw += (self.alpha / n_samples) * (l1_term + l2_term)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * float(db)

                if has_tqdm and epoch % max(1, self.epochs // 10) == 0:
                    mse = np.mean(error**2)
                    iterator.set_postfix({"mse": f"{mse:.4f}"})

                if np.max(np.abs(self.learning_rate * dw)) < self.tolerance:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return np.dot(X, self.weights) + self.bias
