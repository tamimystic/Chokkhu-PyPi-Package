from __future__ import annotations

import numpy as np

from chokkhu.core.logger import Logger

from .base import ChokkhuModel
from .ml import KNN, KMeans, LinearRegression, LogisticRegression, NaiveBayes


def train(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray | None = None,
    task: str = "auto",
    random_state: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> ChokkhuModel:
    if verbose:
        Logger.info(f"Training model: {model} (task: {task})")

    model_obj: ChokkhuModel | None = None

    if model == "linear_regression":
        model_obj = LinearRegression(**kwargs)
    elif model == "ridge":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="ridge", **kwargs
        )
    elif model == "lasso":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="lasso", **kwargs
        )
    elif model == "elastic_net":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="elastic_net", **kwargs
        )
    elif model == "logistic_regression":
        model_obj = LogisticRegression(**kwargs)
    elif model == "knn":
        model_obj = KNN(task=task if task != "auto" else "classification", **kwargs)
    elif model == "naive_bayes":
        model_obj = NaiveBayes(**kwargs)
    elif model == "kmeans":
        model_obj = KMeans(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Model {model} is not supported yet.")

    model_obj.fit(X_train, y_train)

    if verbose:
        Logger.info(f"Successfully trained {model}")

    return model_obj
