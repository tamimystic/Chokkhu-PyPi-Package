import pandas as pd


class MetadataAnalyzer:
    """
    Topic 1: Metadata & Structural EDA
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        results = {}

        # 1. Shape & Volume Inspection
        results["rows"] = df.shape[0]
        results["cols"] = df.shape[1]

        # 2. Data Type Profiling
        dtypes_counts = df.dtypes.value_counts().to_dict()
        results["dtype_profiling"] = {str(k): v for k, v in dtypes_counts.items()}

        # 3. Memory Footprint Analysis
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        results["memory_kb"] = memory_usage_bytes / 1024
        results["memory_mb"] = results["memory_kb"] / 1024

        # 4. Index & Primary Key Integrity
        results["index_duplicates"] = df.index.duplicated().sum()
        results["index_has_nulls"] = df.index.isnull().sum()

        # 5. File Encoding & Sanity Check (String Columns)
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        corrupted_chars = ["\ufffd", "?", "*"]
        encoding_issues = {}
        for col in obj_cols:
            sample = df[col].dropna().astype(str).head(1000)
            issues = sum(sample.apply(lambda x: any(c in x for c in corrupted_chars)))
            if issues > 0:
                encoding_issues[col] = issues
        results["potential_encoding_issues"] = encoding_issues

        return results
