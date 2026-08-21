# Data Splitting (`ck.split`)

Chokkhu provides multi-way stratified partitioning and cross-validation generators.

## Syntax

```python
import chokkhu as ck

X_train, X_test, y_train, y_test = ck.split(
    data=df,
    target="target_column",
    test_size=0.2,
    stratify=True,
    random_state=42
)
```

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` / `dict`| Required | The input data structure. |
| `target` | `str` | Required | The target column or key. |
| `test_size` | `float` | `0.2` | The proportion of the dataset to include in the test split. |
| `val_size` | `float` | `0.0` | If greater than `0.0`, a 3-way split (Train/Val/Test) is returned. |
| `stratify` | `bool` | `False` | If True, data is split in a stratified fashion based on the target labels. |
| `random_state` | `int` | `None` | Seed used by the random number generator for reproducibility. |
| `method` | `str` | `"train_test"`| Determines split mode. Options: `"train_test"`, `"kfold"`, `"stratified_kfold"`, `"timeseries"`. |

??? example "Cross Validation Generator Example"
    ```python
    for fold, (train_df, val_df) in enumerate(ck.split(df, method="kfold", n_splits=5)):
        print(f"Fold {fold}: Train size {len(train_df)}")
    ```
