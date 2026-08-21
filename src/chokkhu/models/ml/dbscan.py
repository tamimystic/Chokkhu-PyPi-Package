from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel


class DBSCAN(ChokkhuModel):
    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> DBSCAN:
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)
        cluster_id = 0

        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

        return self

    def _expand_cluster(
        self,
        X: np.ndarray,
        point_idx: int,
        neighbors: list[int],
        cluster_id: int,
        visited: np.ndarray,
    ) -> None:
        if self.labels_ is None:
            return

        self.labels_[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1

    def _region_query(self, X: np.ndarray, point_idx: int) -> list[int]:
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DBSCAN does not support prediction for new samples. Use .labels_ instead."
        )
