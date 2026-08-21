# Exploratory Data Analysis (`ck.eda`)

Chokkhu provides powerful, automated EDA engines for both tabular and image datasets.

## Tabular EDA

Generates univariate, bivariate, and multivariate statistical reports with interactive visualizations (HTML) and inline notebook rendering.

### Syntax
```python
import chokkhu as ck

ck.eda.tabular("data.csv", target_col="target_column", save_reports=True)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dataset_path` | `str` or `pd.DataFrame` | Required | The path to the dataset or the DataFrame object itself. |
| `target_col` | `str` | `None` | The target feature. If provided, bivariate analysis will compute IV, WoE, and T-Tests. |
| `save_reports` | `bool` | `False` | If True, saves all figures and generates an HTML report. |
| `save_dir` | `str` | `"./chokkhu_outputs/tabular_reports"` | Directory to save the generated reports. |

---

## Image EDA

Analyzes resolution distributions, color spaces, blur scores (Laplacian variance), Shannon entropy, SNR, and edge intensity.

### Syntax
```python
ck.eda.image("dataset_images/", save_reports=True)
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dataset_path` | `str` | Required | The path to the root directory containing class subdirectories of images. |
| `save_reports` | `bool` | `False` | If True, saves all metric figures and generates an HTML report. |
| `save_dir` | `str` | `"./chokkhu_outputs/image_reports"` | Directory to save the generated reports. |
