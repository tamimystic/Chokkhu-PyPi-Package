from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class KNN(ChokkhuModel):
    def __init__(self, n_neighbors: int = 5, task: str = "classification"):
        self.n_neighbors = n_neighbors
        self.task = task
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KNN:
        if y is None:
            raise ValueError("y cannot be None for KNN")
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet.")

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]

            if self.task == "classification":
                labels, counts = np.unique(k_nearest_labels, return_counts=True)
                most_common = labels[np.argmax(counts)]
                predictions.append(most_common)
            else:
                predictions.append(np.mean(k_nearest_labels))

        return np.array(predictions)
