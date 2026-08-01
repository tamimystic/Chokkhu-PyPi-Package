import numpy as np
import pandas as pd


class CategoricalAnalyzer:
    """
    Topic 4: Qualitative/Categorical Data EDA
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}
        cat_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
        results["categorical_cols"] = cat_cols

        if not cat_cols:
            return results

        # 1. Frequency & Diversity Profiling
        freq_stats = {}
        total_rows = df.shape[0]

        for col in cat_cols:
            series = df[col].astype(str)
            value_counts = series.value_counts()

            # High Cardinality Screening
            unique_count = len(value_counts)
            cardinality_ratio = unique_count / total_rows
            is_high_cardinality = cardinality_ratio > 0.5 or unique_count > 100

            # Rare Label / Long-Tail Detection (labels < 5% frequency)
            rare_labels = value_counts[value_counts / total_rows < 0.05].index.tolist()

            # Shannon Entropy for Categories
            probs = value_counts / total_rows
            entropy = -np.sum(probs * np.log2(probs + 1e-9))

            freq_stats[col] = {
                "unique_count": unique_count,
                "is_high_cardinality": is_high_cardinality,
                "rare_labels_count": len(rare_labels),
                "entropy": entropy,
            }
        results["frequency_stats"] = freq_stats

        # 2. Consistency & Order Check
        consistency = {}
        for col in cat_cols:
            series = df[col].dropna().astype(str)
            if series.empty:
                continue

            # String Inconsistency Screening
            # Check for trailing spaces or case differences
            stripped = series.str.strip()
            lower_cased = stripped.str.lower()

            inconsistencies = 0
            if len(series.unique()) != len(lower_cased.unique()):
                inconsistencies += 1  # Case mismatch detected

            consistency[col] = {"potential_case_spacing_issues": inconsistencies > 0}

        results["consistency"] = consistency
        return results
