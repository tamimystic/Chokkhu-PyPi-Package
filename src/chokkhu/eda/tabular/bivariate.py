from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from chokkhu.eda.tabular.univariate import UnivariateAnalyzer


class BivariateAnalyzer:
    """
    Phase 3: Bivariate Analysis
    Handles Feature vs Feature and Feature vs Target relationships.
    """

    @staticmethod
    def calculate_iv_woe(df: pd.DataFrame, feature: str, target: str) -> Dict[str, Any]:
        df_temp = df[[feature, target]].dropna()
        if df_temp.empty or df_temp[target].nunique() != 2:
            return {}

        # Ensure target is 0/1
        t_max = df_temp[target].max()
        df_temp["target_bin"] = np.where(df_temp[target] == t_max, 1, 0)

        # Bin feature if continuous
        if (
            pd.api.types.is_numeric_dtype(df_temp[feature])
            and df_temp[feature].nunique() > 10
        ):
            df_temp["feature_bin"] = pd.qcut(
                df_temp[feature], q=10, duplicates="drop"
            ).astype(str)
        else:
            df_temp["feature_bin"] = df_temp[feature].astype(str)

        grouped = df_temp.groupby("feature_bin")["target_bin"].agg(["count", "sum"])
        grouped["events"] = grouped["sum"]
        grouped["non_events"] = grouped["count"] - grouped["sum"]

        total_events = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()

        if total_events == 0 or total_non_events == 0:
            return {}

        grouped["event_rate"] = grouped["events"] / total_events
        grouped["non_event_rate"] = grouped["non_events"] / total_non_events

        # Replace 0 with a small epsilon to avoid inf
        grouped["event_rate"] = grouped["event_rate"].replace(0, 0.0001)
        grouped["non_event_rate"] = grouped["non_event_rate"].replace(0, 0.0001)

        grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
        grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped[
            "woe"
        ]

        return {"iv": grouped["iv"].sum(), "woe_dict": grouped["woe"].to_dict()}

    @staticmethod
    def analyze(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        types = UnivariateAnalyzer.infer_data_types(df)

        categorical = types["categorical"]["ordinal"] + types["categorical"]["nominal"]
        numerical = types["numerical"]["discrete"] + types["numerical"]["continuous"]

        # ---------------------------------------------------------
        # 1. Feature vs Target (Predictive Power)
        # ---------------------------------------------------------
        target_analysis = {}
        iv_results = {}
        if target_col and target_col in df.columns:
            # Check if target is binary
            if df[target_col].nunique() == 2:
                for col in df.columns:
                    if col == target_col:
                        continue
                    iv_data = BivariateAnalyzer.calculate_iv_woe(df, col, target_col)
                    if iv_data and "iv" in iv_data:
                        iv_results[col] = iv_data["iv"]

            target_analysis["information_value"] = iv_results

            # ANOVA / Kruskal (Categorical vs Continuous Target)
            if target_col in numerical:
                anova_res = {}
                for cat_col in categorical:
                    groups = [
                        group[target_col].dropna().values
                        for name, group in df.groupby(cat_col)
                        if len(group) > 5
                    ]
                    if len(groups) >= 2:
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            h_stat, kp_val = stats.kruskal(*groups)
                            anova_res[cat_col] = {"anova_p": p_val, "kruskal_p": kp_val}
                        except Exception:
                            pass
                target_analysis["categorical_vs_target_anova"] = anova_res

        results["target_analysis"] = target_analysis
        return results
