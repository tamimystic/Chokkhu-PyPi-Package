from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class SVM(ChokkhuModel):
    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        epochs: int = 1000,
    ) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SVM:
        if y is None:
            raise ValueError("y cannot be None for SVM")

        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError(
                "Currently only binary classification is supported for SVM"
            )

        y_ = np.where(y == self.classes[0], -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * float(-y_[idx])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.classes is None:
            raise ValueError("Model is not fitted yet.")
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        return np.where(predictions == -1, self.classes[0], self.classes[1])
