import re

import pandas as pd


class SpecializedAnalyzer:
    """
    Topic 6: Specialized Columns EDA (Datetime, Text, Spatial)
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}

        # 1. Date-Time EDA
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        # Also try to auto-detect object columns that might be dates if they have 'date' in name
        for col in df.select_dtypes(include=["object"]).columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        dt_cols.append(col)
                except Exception:
                    pass

        dt_cols = list(set(dt_cols))
        results["datetime_cols"] = dt_cols

        dt_stats = {}
        for col in dt_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            # Time gaps (assuming sorted)
            sorted_series = series.sort_values()
            diffs = sorted_series.diff().dt.total_seconds()

            dt_stats[col] = {
                "min_date": str(series.min()),
                "max_date": str(series.max()),
                "unique_dates": series.nunique(),
                "max_gap_seconds": diffs.max() if not diffs.empty else 0,
            }
        results["datetime_stats"] = dt_stats

        # 2. Text/NLP Specific Tabular EDA
        text_cols = []
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            # Assume text if mean length is > 50 chars
            lengths = df[col].dropna().astype(str).str.len()
            if not lengths.empty and lengths.mean() > 50:
                text_cols.append(col)

        results["text_cols"] = text_cols
        text_stats = {}
        for col in text_cols:
            series = df[col].dropna().astype(str)
            char_counts = series.str.len()
            word_counts = series.apply(lambda x: len(re.findall(r"\w+", x)))

            text_stats[col] = {
                "mean_char_length": char_counts.mean(),
                "max_char_length": char_counts.max(),
                "mean_word_count": word_counts.mean(),
                "max_word_count": word_counts.max(),
            }
        results["text_stats"] = text_stats

        # 3. Spatial/Geographical EDA
        spatial_stats = {}
        lat_cols = [c for c in df.columns if "lat" in c.lower()]
        lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

        if lat_cols and lon_cols:
            lat = lat_cols[0]
            lon = lon_cols[0]
            spatial_stats["bounding_box"] = {
                "min_lat": df[lat].min(),
                "max_lat": df[lat].max(),
                "min_lon": df[lon].min(),
                "max_lon": df[lon].max(),
            }
        results["spatial_stats"] = spatial_stats

        return results
