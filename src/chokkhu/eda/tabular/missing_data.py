import numpy as np
import pandas as pd


class MissingDataAnalyzer:
    """
    Topic 2: Missing Data & Imputation Impact EDA
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}

        # 1. Missing Value Density Check
        total_rows = df.shape[0]
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / total_rows) * 100
        results["missing_density"] = missing_pct[missing_pct > 0].to_dict()
        results["total_missing_cells"] = missing_counts.sum()

        # 2 & 3. Null Correlation Matrix
        # Calculate correlation between missingness of different columns
        cols_with_missing = missing_counts[missing_counts > 0].index
        if len(cols_with_missing) > 1:
            null_mask = df[cols_with_missing].isnull().astype(int)
            # Use pandas corr, filling NaNs with 0 if variance is 0
            null_corr = null_mask.corr().fillna(0)
            results["null_correlation"] = null_corr
        else:
            results["null_correlation"] = None

        # 4. Zero-Inflation Detection
        # Check numerical columns for high percentage of zeros
        num_cols = df.select_dtypes(include=[np.number]).columns
        zero_inflation = {}
        for col in num_cols:
            zero_count = (df[col] == 0).sum()
            zero_pct = (zero_count / total_rows) * 100
            if zero_pct > 20:  # arbitrary threshold for warning
                zero_inflation[col] = zero_pct
        results["zero_inflation"] = zero_inflation

        # 5. Imputation Variance Shift Check
        # Compare Mean and Variance before and after mean imputation
        imputation_shift = {}
        for col in cols_with_missing:
            if col in num_cols:
                original_mean = df[col].mean()
                original_var = df[col].var()

                # Impute with mean
                imputed = df[col].fillna(original_mean)
                imputed_mean = imputed.mean()
                imputed_var = imputed.var()

                shift_pct = 0
                if original_var != 0 and not pd.isna(original_var):
                    shift_pct = abs((imputed_var - original_var) / original_var) * 100

                imputation_shift[col] = {
                    "original_mean": original_mean,
                    "imputed_mean": imputed_mean,
                    "original_var": original_var,
                    "imputed_var": imputed_var,
                    "variance_shift_pct": shift_pct,
                }
        results["imputation_variance_shift"] = imputation_shift

        return results
