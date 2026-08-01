import pandas as pd


class TabularStats:
    @staticmethod
    def extract(df: pd.DataFrame) -> dict:
        """Extracts basic statistical metadata and correlation."""
        results = {
            "shape": df.shape,
            "dtypes": df.dtypes,
            "missing": df.isnull().sum(),
            "numerical_cols": df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist(),
            "categorical_cols": df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }
        if len(results["numerical_cols"]) > 0:
            results["correlation"] = df[results["numerical_cols"]].corr()
        else:
            results["correlation"] = pd.DataFrame()

        return results
