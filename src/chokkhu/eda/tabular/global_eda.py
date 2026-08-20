from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class GlobalAnalyzer:

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        results["memory_mb"] = mem_usage
        results["duplicated_rows"] = int(df.duplicated().sum())
        dtype_counts = df.dtypes.value_counts().to_dict()
        results["dtype_profiling"] = {str(k): v for k, v in dtype_counts.items()}
        index_stats = {"is_unique": df.index.is_unique, "has_nans": df.index.hasnans}
        results["index_integrity"] = index_stats
        missing_counts = df.isnull().sum()
        missing_pct = missing_counts / len(df) * 100
        missing_density = missing_pct[missing_pct > 0].to_dict()
        results["missing_density"] = missing_density
        if missing_density:
            results["missing_matrix"] = df[list(missing_density.keys())].isnull()
        else:
            results["missing_matrix"] = pd.DataFrame()
        null_corr = None
        if len(missing_density) > 1:
            missing_df = df[list(missing_density.keys())].isnull().astype(int)
            null_corr = missing_df.corr()
        results["null_correlation"] = null_corr
        zero_inflation = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                zero_inflation[col] = zeros / len(df) * 100
        results["zero_inflation"] = zero_inflation
        imputation_shift = {}
        for col in missing_density.keys():
            if col in num_cols:
                series = df[col].dropna()
                if len(series) < 2:
                    continue
                orig_var = series.var()
                mean_val = series.mean()
                median_val = series.median()
                var_after_mean = df[col].fillna(mean_val).var()
                var_after_median = df[col].fillna(median_val).var()
                imputation_shift[col] = {
                    "original_variance": orig_var,
                    "variance_after_mean": var_after_mean,
                    "variance_after_median": var_after_median,
                    "shift_pct_mean": (
                        abs(orig_var - var_after_mean) / orig_var * 100
                        if orig_var != 0
                        else 0
                    ),
                }
        results["imputation_shift"] = imputation_shift
        imputation_recommendations = {}
        for col, pct in missing_density.items():
            if pct > 30:
                imputation_recommendations[col] = "Drop column (missing > 30%)"
            elif col in num_cols:
                if col in imputation_shift:
                    shift = imputation_shift[col]
                    if shift["variance_after_median"] <= shift["variance_after_mean"]:
                        imputation_recommendations[col] = "Impute with Median"
                    else:
                        imputation_recommendations[col] = "Impute with Mean"
                else:
                    imputation_recommendations[col] = "Impute with Median"
            else:
                imputation_recommendations[col] = "Impute with Mode"
        results["imputation_recommendations"] = imputation_recommendations
        return results
