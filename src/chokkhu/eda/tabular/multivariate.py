from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from chokkhu.eda.tabular.univariate import UnivariateAnalyzer


class MultivariateAnalyzer:
    """
    Phase 4: Multivariate Analysis
    Handles multi-feature interactions, correlations, PCA, MI, and Dataset Drift.
    """

    @staticmethod
    def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        if n == 0:
            return 0.0
        r, k = confusion_matrix.shape
        min_dim = min(k - 1, r - 1)
        if min_dim == 0:
            return 0.0
        return np.sqrt(chi2 / (n * min_dim))

    @staticmethod
    def _mutual_information(x: pd.Series, y: pd.Series) -> float:
        c_xy = pd.crosstab(x, y)
        p_xy = c_xy / c_xy.sum().sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        mi = 0.0
        for i in p_xy.index:
            for j in p_xy.columns:
                if p_xy.loc[i, j] > 0:
                    mi += p_xy.loc[i, j] * np.log2(p_xy.loc[i, j] / (p_x[i] * p_y[j]))
        return mi

    @staticmethod
    def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        # Population Stability Index between two series
        if pd.api.types.is_numeric_dtype(expected):
            bins = pd.qcut(expected, q=buckets, retbins=True, duplicates="drop")[1]
            exp_counts = pd.cut(expected, bins=bins).value_counts(normalize=True)
            act_counts = pd.cut(actual, bins=bins).value_counts(normalize=True)
        else:
            exp_counts = expected.value_counts(normalize=True)
            act_counts = actual.value_counts(normalize=True)

        psi = 0.0
        for idx in exp_counts.index:
            e_val = exp_counts.get(idx, 0.0001)
            a_val = act_counts.get(idx, 0.0001)
            if e_val == 0:
                e_val = 0.0001
            if a_val == 0:
                a_val = 0.0001
            psi += (e_val - a_val) * np.log(e_val / a_val)
        return psi

    @staticmethod
    def analyze(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        types = UnivariateAnalyzer.infer_data_types(df)

        numerical = types["numerical"]["discrete"] + types["numerical"]["continuous"]
        categorical = types["categorical"]["ordinal"] + types["categorical"]["nominal"]

        # ---------------------------------------------------------
        # 3.1 Numerical Correlations & VIF
        # ---------------------------------------------------------
        if len(numerical) > 1:
            num_df = df[numerical].dropna()
            if not num_df.empty:
                print("   >>> 3.1 Correlation Analysis (Pearson, Spearman)")
                # Pearson & Spearman
                results["pearson_corr"] = num_df.corr(method="pearson")
                results["spearman_corr"] = num_df.corr(method="spearman")

                print("   >>> 3.2 Multicollinearity Analysis (VIF)")
                # VIF (Scratch Implementation using inverse correlation matrix)
                corr_matrix = num_df.corr().values
                try:
                    inv_corr = np.linalg.inv(corr_matrix)
                    vif = np.diagonal(inv_corr)
                    results["vif"] = dict(zip(num_df.columns, vif))
                except np.linalg.LinAlgError:
                    results["vif"] = {}

        # ---------------------------------------------------------
        # 3.3 Categorical Associations (Cramer's V & MI)
        # ---------------------------------------------------------
        print("   >>> 3.3 Association Analysis (Cramer's V, Mutual Information)")
        if len(categorical) > 1:
            cat_df = df[categorical].dropna()
            cramers_matrix = pd.DataFrame(
                np.ones((len(categorical), len(categorical))),
                index=categorical,
                columns=categorical,
            )

            for i, col1 in enumerate(categorical):
                for j, col2 in enumerate(categorical):
                    if i < j:
                        confusion = pd.crosstab(cat_df[col1], cat_df[col2])
                        cv = MultivariateAnalyzer._cramers_v(confusion)
                        cramers_matrix.loc[col1, col2] = cv
                        cramers_matrix.loc[col2, col1] = cv
            results["cramers_v_matrix"] = cramers_matrix

        if target_col and target_col in df.columns:
            mi_results = {}
            # We bin numeric features to 10 bins to calculate MI
            for col in df.columns:
                if col == target_col:
                    continue

                x = df[col]
                if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
                    x = pd.qcut(x, q=10, duplicates="drop").astype(str)
                else:
                    x = x.astype(str)

                y = df[target_col].astype(str)
                valid = pd.DataFrame({"x": x, "y": y}).dropna()
                if not valid.empty:
                    mi_results[col] = MultivariateAnalyzer._mutual_information(
                        valid["x"], valid["y"]
                    )
            results["mutual_information"] = mi_results

        # ---------------------------------------------------------
        # 3.4 PCA (Deep Embedding)
        # ---------------------------------------------------------
        print("   >>> 3.4 Dimensionality Reduction (PCA)")
        if len(numerical) >= 2:
            num_df = df[numerical].dropna()
            if len(num_df) > 5:
                # Standardize
                standardized = (num_df - num_df.mean()) / (num_df.std() + 1e-9)
                cov_matrix = np.cov(standardized.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                # Sort by eigenvalues
                idx = eigenvalues.argsort()[::-1]
                eigenvectors = eigenvectors[:, idx]
                # Project to 2D
                pca_2d = standardized.dot(eigenvectors[:, :2])
                results["pca_1"] = pca_2d.iloc[:, 0].tolist()
                results["pca_2"] = pca_2d.iloc[:, 1].tolist()
                results["pca_index"] = pca_2d.index.tolist()

        # ---------------------------------------------------------
        # 3.5 Dataset Drift (PSI Approximation via Random Split)
        # ---------------------------------------------------------
        print("   >>> 3.5 Data Drift & Stability (PSI)")
        # For EDA on a single dataset, we simulate drift by splitting 50/50 randomly
        drift_results = {}
        if len(df) > 100:
            df_shuffled = df.sample(frac=1, random_state=42)
            half = len(df_shuffled) // 2
            expected_df = df_shuffled.iloc[:half]
            actual_df = df_shuffled.iloc[half:]

            for col in df.columns:
                try:
                    drift_results[col] = MultivariateAnalyzer._psi(
                        expected_df[col].dropna(), actual_df[col].dropna()
                    )
                except Exception:
                    pass
        results["dataset_drift_psi"] = drift_results

        return results
