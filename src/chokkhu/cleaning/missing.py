from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _knn_impute_col(df: pd.DataFrame, col: str, k: int = 5) -> pd.Series:
    s = df[col].copy()
    missing_mask = s.isna().to_numpy()
    if not missing_mask.any():
        return s
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != col]
    if not num_cols:
        return s.fillna(s.median())
    complete_mask = ~df[num_cols].isna().any(axis=1).to_numpy() & ~missing_mask
    if not complete_mask.any():
        return s.fillna(s.median())
    X_complete = np.array(
        df[num_cols].to_numpy(dtype=np.float64)[complete_mask], copy=True
    )
    y_complete = np.array(s.to_numpy(dtype=np.float64)[complete_mask], copy=True)
    s_arr = np.array(s.to_numpy(dtype=np.float64), copy=True)
    X_all = np.array(df[num_cols].to_numpy(dtype=np.float64), copy=True)
    missing_indices = np.where(missing_mask)[0]
    for idx in missing_indices:
        row_feat = X_all[idx]
        if np.isnan(row_feat).any():
            s_arr[idx] = float(np.nanmean(y_complete))
            continue
        dists = np.linalg.norm(X_complete - row_feat, axis=1)
        k_nearest = np.argsort(dists)[: min(k, len(dists))]
        weights = 1.0 / (dists[k_nearest] + 1e-08)
        s_arr[idx] = float(np.average(y_complete[k_nearest], weights=weights))
    return pd.Series(s_arr, index=df.index, name=col)


def _iterative_impute(df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
    df_res = df.copy()
    num_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
    missing_cols = [c for c in num_cols if df_res[c].isna().any()]
    if not missing_cols:
        return df_res
    for c in num_cols:
        df_res[c] = df_res[c].fillna(df_res[c].median())
    for _ in range(max_iter):
        for c in missing_cols:
            orig_missing = df[c].isna().to_numpy()
            predictors = [p for p in num_cols if p != c]
            if not predictors:
                continue
            X_train = df_res.loc[~orig_missing, predictors].to_numpy(dtype=np.float64)
            y_train = df_res.loc[~orig_missing, c].to_numpy(dtype=np.float64)
            X_missing = df_res.loc[orig_missing, predictors].to_numpy(dtype=np.float64)
            if len(X_train) == 0 or len(X_missing) == 0:
                continue
            X_train_b = np.column_stack([np.ones(len(X_train)), X_train])
            X_missing_b = np.column_stack([np.ones(len(X_missing)), X_missing])
            w, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
            df_res.loc[orig_missing, c] = X_missing_b @ w
    return df_res


def handle_missing(
    data: pd.DataFrame,
    strategy: str = "median",
    threshold: float = 0.5,
    fill_value: object = 0,
    knn_k: int = 5,
    interpolate_method: str = "linear",
    interpolate_order: int = 2,
    iterative_max_iter: int = 10,
) -> pd.DataFrame:
    df = data.copy()
    if strategy is None:
        return df
    if strategy == "drop_cols":
        return df.loc[:, df.isna().mean() <= threshold]
    if strategy == "drop_rows":
        return df.dropna()
    if strategy == "iterative":
        return _iterative_impute(df, max_iter=iterative_max_iter)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in df.columns:
        if not df[col].isna().any():
            continue
        pct = df[col].isna().mean()
        if strategy == "auto":
            if pct > threshold:
                df = df.drop(columns=[col])
                continue
            if col in num_cols:
                series = df[col].dropna()
                is_normal = False
                if len(series) >= 8:
                    try:
                        _, p = stats.shapiro(series[:5000])
                        is_normal = p > 0.05
                    except Exception:
                        is_normal = False
                val = series.mean() if is_normal else series.median()
                df[col] = df[col].fillna(val)
            else:
                mode_val = (
                    df[col].mode().iloc[0] if not df[col].mode().empty else "missing"
                )
                df[col] = df[col].fillna(mode_val)
        elif strategy == "mean" and col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = (
                df[col].mode().iloc[0] if not df[col].mode().empty else fill_value
            )
            df[col] = df[col].fillna(mode_val)
        elif strategy == "constant":
            df[col] = df[col].fillna(fill_value)
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "interpolate" and col in num_cols:
            if interpolate_method in ["polynomial", "spline"]:
                df[col] = df[col].interpolate(
                    method=interpolate_method, order=interpolate_order
                )
            else:
                df[col] = df[col].interpolate(method="linear")
        elif strategy == "knn" and col in num_cols:
            df[col] = _knn_impute_col(df, col, k=knn_k)
        elif col in cat_cols:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "missing"
            df[col] = df[col].fillna(mode_val)
    return df
