<div align="center">

<img src="https://raw.githubusercontent.com/tamimystic/chokkhu/main/profile.jpg" width="140" height="140" style="border-radius:50%;" alt="Author Profile">

# Chokkhu

**An End-to-End, Research-Grade ML and Data Science Pipeline Toolkit for Tabular and Computer Vision Datasets.**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/chokkhu/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/chokkhu/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/tamimystic/chokkhu/blob/main/LICENSE)

> "Minimalistic Code. Maximum Output. Zero Heavy Dependencies."

</div>

---

## Overview

Chokkhu is a high-performance Python toolkit designed to streamline the complete Machine Learning lifecycle including raw data loading, statistical Exploratory Data Analysis, advanced data cleaning, preprocessing, feature transformation, and stratified data splitting.

### Core Philosophy and Design Principles
1. **Zero Heavy Dependencies**: Built from the ground up using pure NumPy, Pandas, SciPy, Matplotlib, Seaborn, and OpenCV-headless. No Torch, TensorFlow, Scikit-Learn, or imbalanced-learn dependencies required.
2. **Unified Minimalist API**: Powerful operations encapsulated in intuitive, one-line functions (load, eda, clean, preprocess, transform, split).
3. **Cross-Platform Compatibility**: Tested and verified across 10 environments (Python 3.8 to 3.12 on Linux Ubuntu and Windows).

---

## Installation

Install Chokkhu directly from PyPI via pip:

```bash
pip install --upgrade chokkhu
```

---

## Complete End-to-End Pipeline in One Workflow

```python
import chokkhu as ck

df = ck.load("dataset.csv")

ck.eda.tabular("dataset.csv", target_col="price", save_reports=True)

df_clean = ck.clean(
    df,
    missing="knn",
    outliers="isolation_forest",
    duplicates=True,
    fix_data_types=True
)

df_proc, state = ck.preprocess(
    df_clean,
    target="price",
    scale="standard",
    encode="onehot",
    select_features="correlation"
)

df_trans = ck.transform(
    df_proc,
    target="price",
    resample="smote",
    pca=10
)

X_tr, X_val, X_te, y_tr, y_val, y_te = ck.split(
    df_trans,
    target="price",
    test_size=0.2,
    val_size=0.1,
    stratify=True,
    random_state=42
)
```

---

## Detailed API Reference

### 1. Data Loading (`chokkhu.load`)
Auto-detects file format and loads tabular or image datasets seamlessly:

```python
import chokkhu as ck

df = ck.load("data.csv")

images_data = ck.load("images_folder/", data_type="image", target_size=(128, 128))
```

---

### 2. Exploratory Data Analysis (`chokkhu.eda`)

#### Tabular EDA
Generates univariate, bivariate, and multivariate statistical reports with interactive visualizations:

```python
ck.eda.tabular(
    dataset_path="data.csv",
    target_col="target",
    save_reports=True,
    save_dir="./eda_reports"
)
```

#### Image EDA
Analyzes resolution distributions, color spaces, blur scores (Laplacian variance), pure NumPy Shannon entropy, SNR, edge intensity, and perceptual duplicates:

```python
ck.eda.image(
    dataset_path="dataset_images/",
    save_reports=True,
    save_dir="./image_reports"
)
```

---

### 3. Data Cleaning (`chokkhu.clean`)
All-in-one data sanitation function:

```python
df_cleaned = ck.clean(
    data=df,
    missing="knn",
    missing_threshold=0.5,
    outliers="iqr",
    outlier_action="remove",
    duplicates=True,
    fix_data_types=True
)
```

---

### 4. Data Preprocessing (`chokkhu.preprocess`)
Stateful feature scalers, encoders, and feature selection:

```python
df_processed, preprocessor_state = ck.preprocess(
    data=df_cleaned,
    target="target_column",
    scale="standard",
    encode="onehot",
    select_features="mutual_info",
    top_k_features=15
)
```

---

### 5. Data Transformation (`chokkhu.transform`)
Dimensionality reduction, class imbalance resampling, polynomial feature engineering, and image augmentations:

```python
df_transformed = ck.transform(
    data=df_processed,
    target="target_column",
    pca=5,
    lda=2,
    tsne=2,
    resample="smote",
    polynomial=2,
    log_features=["skewed_col"]
)

aug_images_dict = ck.transform(
    data=images_data,
    augment=True,
    augment_techniques=["horizontal_flip", "rotate", "brightness", "contrast", "noise", "crop", "blur", "cutout"],
    augment_factor=3
)
```

---

### 6. Data Splitting (`chokkhu.split`)
Multi-way stratified partitioning and cross-validation generators:

```python
X_train, X_test, y_train, y_test = ck.split(
    df_transformed,
    target="target_column",
    test_size=0.2,
    stratify=True,
    random_state=42
)

X_train, X_val, X_test, y_train, y_val, y_test = ck.split(
    df_transformed,
    target="target_column",
    test_size=0.2,
    val_size=0.1,
    stratify=True,
    random_state=42
)

for fold, (train_df, val_df) in enumerate(ck.split(df, method="kfold", n_splits=5)):
    print(f"Fold {fold}: Train shape={train_df.shape}, Val shape={val_df.shape}")
```

---

## Feature Matrix Summary

| Category | Modules & Algorithms Available |
| :--- | :--- |
| **I/O** | CSV, TSV, JSON, Excel (.xlsx), Parquet, NumPy (.npy, .npz), Image Folders |
| **EDA** | Univariate, Bivariate, Multivariate Correlation, Interactive HTML Reports, Image Quality & Blur Metrics, GLCM Texture, Perceptual Duplicates |
| **Cleaning** | KNN Imputer, MICE (Iterative), Tukey IQR, Isolation Forest, Z-Score, Winsorization, Auto Dtype Fixer |
| **Preprocessing** | Standard, MinMax, Robust, Power, Quantile Scalers; One-Hot, Target, Binary, Frequency Encoders; Variance, Correlation, Mutual Info, RFE Selectors |
| **Transformation** | PCA, SVD, LDA, t-SNE, SMOTE, ADASYN, Tomek Links, Image Augmentation (Flip, Rotation, Brightness, Noise, Crop, Blur, Cutout, Mixup), Polynomial Features |
| **Splitting** | Train/Test, Train/Val/Test (3-way), Stratified Splitting, K-Fold, Stratified K-Fold, TimeSeriesSplit |

---

## Contributing

We welcome community contributions! Feel free to submit issues or pull requests to help expand Chokkhu:

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the **MIT License**. See [`LICENSE`](file:///i:/Inception%20BD/chokkhu/LICENSE) for more details.
