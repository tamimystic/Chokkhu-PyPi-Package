from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class KMeans(ChokkhuModel):
    def __init__(
        self, n_clusters: int = 8, max_iter: int = 300, random_state: int | None = None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> KMeans:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for _ in range(self.max_iter):
            # Assign labels
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # Update centroids
            new_centers = np.array(
                [X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            # Check for convergence
            if np.all(self.cluster_centers_ == new_centers):
                break
            self.cluster_centers_ = new_centers

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
