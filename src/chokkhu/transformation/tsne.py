from __future__ import annotations

from typing import Any

import numpy as np


class TSNE:

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 500,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.maximum(D, 0.0)

    def _h_beta(self, D_row: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
        P = np.exp(-D_row * beta)
        sum_P = float(np.sum(P))
        if sum_P == 0 or np.isnan(sum_P):
            H = 0.0
            P = np.zeros_like(D_row)
        else:
            H = np.log(sum_P) + beta * np.sum(D_row * P) / sum_P
            P = P / sum_P
        return H, P

    def _x2p(self, X: np.ndarray, tol: float = 1e-5) -> np.ndarray:
        n = X.shape[0]
        D = self._pairwise_distances(X)
        P = np.zeros((n, n), dtype=np.float64)
        beta = np.ones(n, dtype=np.float64)
        log_u = np.log(self.perplexity)

        for i in range(n):
            beta_min = -np.inf
            beta_max = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))]
            H, this_p = self._h_beta(Di, beta[i])
            h_diff = H - log_u
            tries = 0
            while np.abs(h_diff) > tol and tries < 50:
                if h_diff > 0:
                    beta_min = beta[i]
                    if np.isinf(beta_max):
                        beta[i] *= 2.0
                    else:
                        beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta_max = beta[i]
                    if np.isinf(beta_min):
                        beta[i] /= 2.0
                    else:
                        beta[i] = (beta[i] + beta_min) / 2.0
                H, this_p = self._h_beta(Di, beta[i])
                h_diff = H - log_u
                tries += 1
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n]))] = this_p

        P = P + P.T
        P = P / float(np.sum(P))
        P = np.maximum(P, 1e-12)
        return P

    def fit_transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        n = X_arr.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        P = self._x2p(X_arr)
        P = P * 4.0

        Y = np.random.randn(n, self.n_components) * 1e-4
        dY = np.zeros((n, self.n_components), dtype=np.float64)
        iY = np.zeros((n, self.n_components), dtype=np.float64)
        gains = np.ones((n, self.n_components), dtype=np.float64)

        for iteration in range(self.n_iter):
            sum_Y = np.sum(np.square(Y), axis=1)
            num = -2.0 * np.dot(Y, Y.T)
            num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.0
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(
                    np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T
                    * (Y[i, :] - Y),
                    axis=0,
                )

            momentum = 0.5 if iteration < 20 else 0.8
            gains = (gains + 0.2) * ((dY > 0.0) != (iY > 0.0)) + (gains * 0.8) * (
                (dY > 0.0) == (iY > 0.0)
            )
            gains = np.maximum(gains, 0.01)
            iY = momentum * iY - self.learning_rate * (gains * dY)
            Y = Y + iY
            Y = Y - np.mean(Y, axis=0)

            if iteration == 100:
                P = P / 4.0

        return Y
