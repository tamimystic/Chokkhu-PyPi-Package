# Data Preprocessing (`ck.preprocess`)

Handles automated feature scaling and categorical encoding. It returns both the processed DataFrame and a `preprocessor_state` object containing the fitted scalers/encoders, allowing you to inverse-transform or apply identical processing to test data later.

## Syntax

```python
import chokkhu as ck

df_proc, state = ck.preprocess(
    data=df,
    target="target_column",
    scale="standard",
    encode="onehot"
)
```

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | Required | The input DataFrame. |
| `target` | `str` | `None` | The target column. This column will be excluded from standard feature scaling. |
| `scale` | `str` | `"standard"` | Scaling strategy. Options: `"standard"`, `"minmax"`, `"robust"`, `"maxabs"`, `None`. |
| `encode` | `str` | `"onehot"` | Encoding strategy for categorical variables. Options: `"onehot"`, `"label"`, `"ordinal"`, `"frequency"`, `None`. |
| `select_features`| `str` | `None` | Feature selection strategy. Options: `"correlation"`, `"mutual_info"`. |
| `top_k_features` | `int` | `10` | The number of top features to retain if `select_features` is active. |

## Returns

- **processed_data** (`pd.DataFrame`): The transformed DataFrame.
- **state** (`dict`): A dictionary containing the fitted instances (e.g., `StandardScaler`, `OneHotEncoder`) used during the transformation.
