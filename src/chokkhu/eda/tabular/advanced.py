import numpy as np
import pandas as pd


class AdvancedAnalyzer:
    """
    Topic 7: Advanced Machine Learning & Target EDA
    """

    @staticmethod
    def _calculate_woe_iv(df, feature, target):
        # Only for categorical feature and binary target (0/1)
        lst = []
        for val in df[feature].unique():
            val_df = df[df[feature] == val]
            good = val_df[target].sum()
            bad = val_df.shape[0] - good
            lst.append({"Value": val, "Good": good, "Bad": bad})

        dset = pd.DataFrame(lst)
        dset["Dist_Good"] = dset["Good"] / (dset["Good"].sum() + 1e-9)
        dset["Dist_Bad"] = dset["Bad"] / (dset["Bad"].sum() + 1e-9)
        dset["WoE"] = np.log((dset["Dist_Good"] / (dset["Dist_Bad"] + 1e-9)) + 1e-9)
        dset["IV"] = (dset["Dist_Good"] - dset["Dist_Bad"]) * dset["WoE"]

        iv = dset["IV"].sum()
        return iv

    @staticmethod
    def _pca_from_scratch(data, n_components=2):
        data = data - np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
        pca_reduced = np.dot(
            eigenvector_subset.transpose(), data.transpose()
        ).transpose()
        return pca_reduced

    @staticmethod
    def analyze(df: pd.DataFrame, target_col: str = None) -> dict:
        results = {}

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. PCA (Principal Component Analysis) from scratch
        if len(num_cols) >= 2:
            try:
                # Impute and standardize
                data_num = df[num_cols].fillna(df[num_cols].median()).values
                data_num = (data_num - np.mean(data_num, axis=0)) / (
                    np.std(data_num, axis=0) + 1e-9
                )
                pca_reduced = AdvancedAnalyzer._pca_from_scratch(
                    data_num, n_components=2
                )
                results["pca_1"] = pca_reduced[:, 0]
                results["pca_2"] = pca_reduced[:, 1]
            except Exception as e:
                results["pca_error"] = str(e)

        # 2. Information Value (IV) & Weight of Evidence (WoE)
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            unique_targets = target_series.unique()

            # If target is binary, calculate IV for categoricals
            if len(unique_targets) == 2 and np.issubdtype(
                target_series.dtype, np.number
            ):
                cat_cols = df.select_dtypes(
                    exclude=[np.number, "datetime"]
                ).columns.tolist()
                if target_col in cat_cols:
                    cat_cols.remove(target_col)

                iv_results = {}
                for col in cat_cols[:10]:  # Limit to 10 to avoid performance issues
                    try:
                        iv = AdvancedAnalyzer._calculate_woe_iv(
                            df.dropna(subset=[col, target_col]), col, target_col
                        )
                        iv_results[col] = iv
                    except Exception:
                        pass
                results["information_value"] = iv_results

        return results
