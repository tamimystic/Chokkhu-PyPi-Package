from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class HierarchicalClustering(ChokkhuModel):
    def __init__(self, n_clusters: int = 2) -> None:
        self.n_clusters = n_clusters
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HierarchicalClustering:
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)

        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])

        np.fill_diagonal(distances, np.inf)

        current_clusters = n_samples

        while current_clusters > self.n_clusters:
            min_idx = np.argmin(distances)
            unraveled = np.unravel_index(min_idx, distances.shape)
            i, j = int(unraveled[0]), int(unraveled[1])

            cluster_i = self.labels_[i]
            cluster_j = self.labels_[j]

            self.labels_[self.labels_ == cluster_j] = cluster_i

            for k in range(n_samples):
                if k != i and k != j:
                    new_dist = min(distances[i, k], distances[j, k])
                    distances[i, k] = new_dist
                    distances[k, i] = new_dist

            distances[j, :] = np.inf
            distances[:, j] = np.inf

            current_clusters -= 1

        unique_labels = np.unique(self.labels_)
        new_labels = np.zeros_like(self.labels_)
        for i, label in enumerate(unique_labels):
            new_labels[self.labels_ == label] = i
        self.labels_ = new_labels

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Hierarchical clustering does not support prediction for new samples. Use .labels_ instead."
        )
