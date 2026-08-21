from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ChokkhuModel


class Node:
    def __init__(
        self,
        feature: int | None = None,
        threshold: float | None = None,
        left: Node | None = None,
        right: Node | None = None,
        *,
        value: float | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree(ChokkhuModel):
    def __init__(
        self,
        task: str = "classification",
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_features: int | None = None,
    ) -> None:
        self.task = task
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> DecisionTree:
        if y is None:
            raise ValueError("y cannot be None for Decision Tree")
        self.n_features = (
            X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        )
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = (
                self._most_common_label(y)
                if self.task == "classification"
                else np.mean(y)
            )
            return Node(value=float(leaf_value))

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None or best_thresh is None:
            leaf_value = (
                self._most_common_label(y)
                if self.task == "classification"
                else np.mean(y)
            )
            return Node(value=float(leaf_value))

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feat_idxs: np.ndarray
    ) -> tuple[int | None, float | None]:
        best_gain = -1.0
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(
        self, y: np.ndarray, X_column: np.ndarray, threshold: float
    ) -> float:
        parent_loss = (
            self._entropy(y) if self.task == "classification" else self._variance(y)
        )

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l = (
            self._entropy(y[left_idxs])
            if self.task == "classification"
            else self._variance(y[left_idxs])
        )
        e_r = (
            self._entropy(y[right_idxs])
            if self.task == "classification"
            else self._variance(y[right_idxs])
        )
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_loss - child_loss
        return ig

    def _split(
        self, X_column: np.ndarray, split_thresh: float
    ) -> tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y: np.ndarray) -> float:
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        entropy = 0.0
        for p in ps:
            if p > 0:
                entropy -= float(p * np.log(p))
        return entropy

    def _variance(self, y: np.ndarray) -> float:
        return float(np.var(y))

    def _most_common_label(self, y: np.ndarray) -> Any:
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.root:
            raise ValueError("Model is not fitted yet.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        if node.is_leaf_node():
            return float(node.value) if node.value is not None else 0.0

        if node.feature is not None and node.threshold is not None:
            if x[node.feature] <= node.threshold:
                if node.left:
                    return self._traverse_tree(x, node.left)
            else:
                if node.right:
                    return self._traverse_tree(x, node.right)
        return 0.0
