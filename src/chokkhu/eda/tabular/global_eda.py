from typing import Any, Dict

import numpy as np
import pandas as pd


class GlobalAnalyzer:
    """
    Phase 1: Global Dataset Profiling
    Handles Metadata, Structural EDA, and Missing Data Analysis.
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # ---------------------------------------------------------
        # 1. Metadata & Structural Profiling
        # ---------------------------------------------------------
        # Shape & Volume
        results["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

        # Memory Footprint
        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        results["memory_mb"] = mem_usage

        # Data Type Profiling
        dtype_counts = df.dtypes.value_counts().to_dict()
        results["dtype_profiling"] = {str(k): v for k, v in dtype_counts.items()}

        # Index & Primary Key Integrity
        index_stats = {
            "is_unique": df.index.is_unique,
            "has_nans": df.index.hasnans,
        }
        results["index_integrity"] = index_stats

        # ---------------------------------------------------------
        # 2. Missing Data Analysis
        # ---------------------------------------------------------
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_density = missing_pct[missing_pct > 0].to_dict()
        results["missing_density"] = missing_density

        # Missingno Matrix Data
        # We'll pass the boolean mask of nulls for plotter to render the matrix
        if missing_density:
            results["missing_matrix"] = df[list(missing_density.keys())].isnull()
        else:
            results["missing_matrix"] = pd.DataFrame()

        # Nullity Correlation
        null_corr = None
        if len(missing_density) > 1:
            missing_df = df[list(missing_density.keys())].isnull().astype(int)
            null_corr = missing_df.corr()
        results["null_correlation"] = null_corr

        # Zero-Inflation Check (High % of exact 0s in numeric cols)
        zero_inflation = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                zero_inflation[col] = (zeros / len(df)) * 100
        results["zero_inflation"] = zero_inflation

        # Imputation Shift Analysis
        # Variance shift if we impute with Mean vs Median
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

        return results
