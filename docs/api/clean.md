# Data Cleaning (`ck.clean`)

An all-in-one data sanitation function to intelligently handle missing values, outliers, duplicate records, and data type inferences.

## Syntax

```python
import chokkhu as ck

df_cleaned = ck.clean(
    data=df,
    missing="knn",
    outliers="iqr",
    duplicates=True,
    fix_data_types=True
)
```

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | Required | The input DataFrame to clean. |
| `missing` | `str` | `"median"` | Missing value imputation strategy. Options: `"mean"`, `"median"`, `"mode"`, `"knn"`, `"iterative"`, `None`. |
| `missing_threshold` | `float` | `0.5` | Drops columns that have a missing ratio greater than this threshold. |
| `knn_k` | `int` | `5` | The number of neighbors to use if `missing="knn"`. |
| `outliers` | `str` | `"iqr"` | Outlier detection strategy. Options: `"iqr"`, `"zscore"`, `"isolation_forest"`, `None`. |
| `outlier_action` | `str` | `"remove"` | What to do with detected outliers. Options: `"remove"`, `"clip"`. |
| `zscore_threshold` | `float`| `3.0` | The Z-score threshold for outlier detection. |
| `duplicates` | `bool` | `True` | If True, duplicate rows are removed. |
| `fix_data_types`| `bool` | `True` | If True, automatically downcasts numeric types and detects datetimes. |

??? example "Advanced Cleaning Configuration"
    ```python
    df_clean = ck.clean(
        data=df,
        missing="iterative",
        iterative_max_iter=20,
        outliers="isolation_forest",
        outlier_action="clip",
        fix_data_types=True
    )
    ```
