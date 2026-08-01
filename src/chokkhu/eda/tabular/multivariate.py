import numpy as np
import pandas as pd
from scipy import stats


class MultivariateAnalyzer:
    """
    Topic 5: Bivariate & Multivariate EDA
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

        # 1. Numerical vs Numerical
        if len(num_cols) > 1:
            # Pearson, Spearman, Kendall
            results["pearson_corr"] = df[num_cols].corr(method="pearson").fillna(0)
            results["spearman_corr"] = df[num_cols].corr(method="spearman").fillna(0)

            # VIF (Variance Inflation Factor) without statsmodels
            # VIF = 1 / (1 - R^2)
            vif_data = {}
            # Fill NaNs with mean just for VIF calculation
            data_num = df[num_cols].fillna(df[num_cols].mean())
            if len(data_num) > 10:  # Need enough rows
                X = data_num.values
                # Standardize
                X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
                for i, col in enumerate(num_cols):
                    y = X[:, i]
                    X_others = np.delete(X, i, axis=1)
                    # Add intercept
                    X_others = np.hstack([np.ones((X_others.shape[0], 1)), X_others])
                    try:
                        # OLS: beta = (X^T X)^-1 X^T y
                        beta = np.linalg.inv(X_others.T @ X_others) @ X_others.T @ y
                        y_pred = X_others @ beta
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        ss_res = np.sum((y - y_pred) ** 2)
                        r_squared = 1 - (ss_res / (ss_tot + 1e-9))
                        vif = 1 / (1 - r_squared + 1e-9)
                        vif_data[col] = vif
                    except np.linalg.LinAlgError:
                        vif_data[col] = float("inf")
                results["vif"] = vif_data

        # 2. Categorical vs Categorical (Chi-Square & Cramer's V)
        cat_cat_results = []
        if len(cat_cols) > 1:
            # Sample up to 5 categorical columns to avoid O(N^2) explosion
            target_cats = cat_cols[:5]
            for i in range(len(target_cats)):
                for j in range(i + 1, len(target_cats)):
                    col1, col2 = target_cats[i], target_cats[j]
                    contingency = pd.crosstab(df[col1], df[col2])
                    if (
                        not contingency.empty
                        and contingency.shape[0] > 1
                        and contingency.shape[1] > 1
                    ):
                        chi2, p, dof, ex = stats.chi2_contingency(contingency)
                        n = contingency.sum().sum()
                        min_dim = min(contingency.shape) - 1
                        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                        cat_cat_results.append(
                            {
                                "feature_1": col1,
                                "feature_2": col2,
                                "chi2": chi2,
                                "p_value": p,
                                "cramers_v": cramers_v,
                            }
                        )
        results["categorical_relations"] = cat_cat_results

        # 3. Numerical vs Categorical (ANOVA / T-Test)
        num_cat_results = []
        if num_cols and cat_cols:
            # Limit to top 5 num and 5 cat to prevent explosion
            for cat_col in cat_cols[:5]:
                unique_vals = df[cat_col].dropna().unique()
                if 1 < len(unique_vals) <= 10:  # Only if groups are small
                    for num_col in num_cols[:5]:
                        groups = [
                            df[df[cat_col] == val][num_col].dropna().values
                            for val in unique_vals
                        ]
                        # Ensure all groups have data
                        groups = [g for g in groups if len(g) > 2]
                        if len(groups) == 2:
                            # T-Test
                            stat, p = stats.ttest_ind(
                                groups[0], groups[1], equal_var=False
                            )
                            test_type = "T-Test"
                        elif len(groups) > 2:
                            # ANOVA
                            stat, p = stats.f_oneway(*groups)
                            test_type = "ANOVA"
                        else:
                            continue

                        # Kruskal-Wallis (Non-parametric)
                        try:
                            _, k_p = stats.kruskal(*groups)
                        except ValueError:
                            _, k_p = np.nan, np.nan

                        num_cat_results.append(
                            {
                                "categorical": cat_col,
                                "numerical": num_col,
                                "test_type": test_type,
                                "stat": stat,
                                "p_value": p,
                                "kruskal_p_value": k_p,
                            }
                        )
        results["numerical_categorical_relations"] = num_cat_results

        return results
