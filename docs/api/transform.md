# Transformation & Augmentation (`ck.transform`)

Handles advanced dataset transformations including dimensionality reduction (PCA, t-SNE), class imbalance resampling (SMOTE), polynomial feature engineering, and image augmentations.

## Syntax (Tabular)

```python
import chokkhu as ck

df_transformed = ck.transform(
    data=df,
    target="target_column",
    pca=10,
    resample="smote"
)
```

## Syntax (Image)

```python
import chokkhu as ck

aug_images = ck.transform(
    data=images_dict,
    augment=True,
    augment_techniques=["horizontal_flip", "rotate", "noise"],
    augment_factor=3
)
```

## Tabular Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | Required | The input DataFrame. |
| `target` | `str` | Required | The target column (needed for resampling). |
| `pca` | `int` | `None` | Applies PCA and retains the specified number of components. |
| `tsne` | `int` | `None` | Applies t-SNE and retains the specified number of components. |
| `resample` | `str` | `None` | Handles class imbalance. Options: `"smote"`, `"adasyn"`, `"over"`, `"under"`. |
| `polynomial` | `int` | `None` | Generates polynomial features up to the specified degree. |

## Image Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `dict` | Required | Dictionary containing `"images"` and `"labels"`. |
| `augment` | `bool` | `False` | Enables image augmentation. |
| `augment_techniques` | `list` | `[]` | List of techniques to apply randomly. Options: `"horizontal_flip"`, `"rotate"`, `"brightness"`, `"noise"`, `"crop"`, `"blur"`, `"cutout"`. |
| `augment_factor` | `int` | `1` | The multiplier for the dataset size (e.g., `2` generates two augmented variants per original image). |
