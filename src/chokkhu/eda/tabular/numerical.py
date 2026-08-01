import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


class NumericalAnalyzer:
    """
    Topic 3: Quantitative/Numerical Data EDA
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        results["numerical_cols"] = num_cols

        if not num_cols:
            return results

        # 1. Descriptive Statistical Profiling
        desc_stats = {}
        for col in num_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            desc_stats[col] = {
                "mean": series.mean(),
                "median": series.median(),
                "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
                "range": series.max() - series.min(),
                "variance": series.var(),
                "std": series.std(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
            }
        results["descriptive_stats"] = desc_stats

        # 2. Distribution Shape & Mathematical Tests
        dist_shape = {}
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            skewness = series.skew()
            kurtosis = series.kurtosis()

            # Normality Hypothesis Testing (Shapiro-Wilk)
            # Shapiro-Wilk is not accurate for N > 5000, so we sample if necessary
            sample = (
                series if len(series) <= 5000 else series.sample(5000, random_state=42)
            )
            try:
                stat, p_value = stats.shapiro(sample)
                is_normal = p_value > 0.05
            except Exception:
                is_normal = False
                p_value = np.nan

            # Multimodal Detection using Gaussian KDE and peak finding
            try:
                kde = stats.gaussian_kde(series)
                x_grid = np.linspace(series.min(), series.max(), 100)
                kde_vals = kde(x_grid)
                peaks, _ = find_peaks(kde_vals)
                modality = (
                    "Unimodal"
                    if len(peaks) <= 1
                    else ("Bimodal" if len(peaks) == 2 else "Multimodal")
                )
            except Exception:
                modality = "Unknown"

            dist_shape[col] = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "is_normal": is_normal,
                "shapiro_p_value": p_value,
                "modality": modality,
            }
        results["distribution_shape"] = dist_shape

        # 3. Outlier & Anomaly Detection
        outliers = {}
        for col in num_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            # Z-Score Method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = (z_scores > 3).sum()

            # Tukey's IQR Method
            q1 = desc_stats[col]["q1"]
            q3 = desc_stats[col]["q3"]
            iqr = desc_stats[col]["iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            tukey_outliers = ((series < lower_bound) | (series > upper_bound)).sum()

            # Hampel Identifier (MAD based)
            median = desc_stats[col]["median"]
            mad = stats.median_abs_deviation(series, scale="normal")
            if mad == 0:
                hampel_outliers = 0
            else:
                hampel_scores = np.abs(series - median) / mad
                hampel_outliers = (hampel_scores > 3).sum()

            outliers[col] = {
                "z_score_outliers": int(z_outliers),
                "tukey_outliers": int(tukey_outliers),
                "hampel_outliers": int(hampel_outliers),
            }

        # Mahalanobis Distance for multivariate outliers
        if len(num_cols) > 1:
            try:
                # Fill NaNs with median for covariance calc
                data_filled = df[num_cols].fillna(df[num_cols].median())
                data_np = data_filled.to_numpy()
                mean_vec = np.mean(data_np, axis=0)
                cov_mat = np.cov(data_np, rowvar=False)
                inv_cov_mat = np.linalg.inv(cov_mat)

                diff = data_np - mean_vec
                left = np.dot(diff, inv_cov_mat)
                mahalanobis = np.sum(left * diff, axis=1)

                # Chi-Square threshold for Mahalanobis
                threshold = stats.chi2.ppf(0.975, df=len(num_cols))
                multi_outliers = (mahalanobis > threshold).sum()
                results["multivariate_outliers_mahalanobis"] = int(multi_outliers)
            except np.linalg.LinAlgError:
                results["multivariate_outliers_mahalanobis"] = "Singular Matrix Error"

        results["univariate_outliers"] = outliers
        return results
