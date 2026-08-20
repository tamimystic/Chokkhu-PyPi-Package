import numpy as np
import pandas as pd
from typing import Tuple, Union, Generator

def train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series] = None,
    test_size: float = 0.2,
    val_size: float = None,
    shuffle: bool = True,
    stratify: bool = False,
    random_state: int = None
):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    
    if stratify and y is not None:
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        train_idx, test_idx, val_idx = [], [], []
        for c in classes:
            c_idx = np.where(y_arr == c)[0]
            if shuffle:
                np.random.shuffle(c_idx)
            n_c = len(c_idx)
            n_test = int(n_c * test_size)
            if val_size is not None:
                n_val = int(n_c * val_size)
                test_idx.extend(c_idx[:n_test])
                val_idx.extend(c_idx[n_test:n_test + n_val])
                train_idx.extend(c_idx[n_test + n_val:])
            else:
                test_idx.extend(c_idx[:n_test])
                train_idx.extend(c_idx[n_test:])
        train_idx, test_idx = np.array(train_idx), np.array(test_idx)
        val_idx = np.array(val_idx) if val_size is not None else None
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        n_test = int(n * test_size)
        if val_size is not None:
            n_val = int(n * val_size)
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]
        else:
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            val_idx = None

    if is_df:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        X_val = X.iloc[val_idx] if val_idx is not None else None
    else:
        X_train, X_test = X[train_idx], X[test_idx]
        X_val = X[val_idx] if val_idx is not None else None

    if y is not None:
        if is_series:
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_val = y.iloc[val_idx] if val_idx is not None else None
        else:
            y_train, y_test = y[train_idx], y[test_idx]
            y_val = y[val_idx] if val_idx is not None else None
        if val_size is not None:
            return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test

    if val_size is not None:
        return X_train, X_val, X_test
    return X_train, X_test

def kfold(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series] = None,
    k: int = 5,
    stratified: bool = False,
    shuffle: bool = True,
    random_state: int = None
) -> Generator:
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    
    if stratified and y is not None:
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        folds = [[] for _ in range(k)]
        for c in classes:
            c_idx = np.where(y_arr == c)[0]
            if shuffle:
                np.random.shuffle(c_idx)
            for i, idx in enumerate(c_idx):
                folds[i % k].append(idx)
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        folds = [indices[i::k] for i in range(k)]

    for i in range(k):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        if is_df:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_tr, X_va = X[train_idx], X[val_idx]
        if y is not None:
            if is_series:
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_tr, y_va = y[train_idx], y[val_idx]
            yield X_tr, X_va, y_tr, y_va
        else:
            yield X_tr, X_va

def time_series_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series] = None,
    n_splits: int = 5
) -> Generator:
    n = len(X)
    fold_size = n // (n_splits + 1)
    is_df = isinstance(X, pd.DataFrame)
    is_series = isinstance(y, (pd.Series, pd.DataFrame))
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_start = train_end
        val_end = min(val_start + fold_size, n)
        if is_df:
            X_tr, X_va = X.iloc[:train_end], X.iloc[val_start:val_end]
        else:
            X_tr, X_va = X[:train_end], X[val_start:val_end]
        if y is not None:
            if is_series:
                y_tr, y_va = y.iloc[:train_end], y.iloc[val_start:val_end]
            else:
                y_tr, y_va = y[:train_end], y[val_start:val_end]
            yield X_tr, X_va, y_tr, y_va
        else:
            yield X_tr, X_va
