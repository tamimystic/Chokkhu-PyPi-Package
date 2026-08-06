<div align="center">

# Chokkhu

**A Professional Deep Learning & Tabular EDA and Preprocessing Toolkit**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![License](https://img.shields.io/github/license/tamimystic/Chokkhu-PyPi-Package.svg?style=for-the-badge)](https://github.com/tamimystic/Chokkhu-PyPi-Package/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/Chokkhu-PyPi-Package/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/Chokkhu-PyPi-Package/actions)

*Industry-grade data preparation, automated EDA, and class balancing for Computer Vision and Tabular datasets.*

</div>

---

## Overview

**Chokkhu** is a high-performance Python package designed to streamline the most tedious parts of Machine Learning: Data Analysis and Preprocessing. 

Whether you are working with large image datasets for Deep Learning (TensorFlow/Keras) or complex Tabular data, Chokkhu automates Exploratory Data Analysis (EDA), missing value handling, class imbalance correction, and generates professional statistical reports.

---

## Installation

Install Chokkhu via pip. TensorFlow and other scientific libraries are automatically installed as runtime dependencies.

```bash
pip install chokkhu
```

> **Note**: Chokkhu works seamlessly in Google Colab, Jupyter Notebooks, and enterprise CI/CD environments.

---

## Quick Start: Exploratory Data Analysis (EDA)

Chokkhu provides a unified, dynamic API for EDA. It automatically detects parameters and adjusts its output.

### 1. Tabular EDA

Generate a comprehensive suite of statistical charts (Univariate, Bivariate, Multivariate, PCA, VIF) to understand your CSV datasets in one simple call:

```python
import chokkhu as ck

ck.eda.tabular(
    dataset_path="dataset.csv", 
    save_dir="reports/tabular",  
    target_col="TargetVariable"  
)
```

**Parameters:**
- `dataset_path` *(str)*: Path to your CSV file.
- `save_dir` *(str, optional)*: If provided, all reports are automatically saved to this directory. If not provided, reports are displayed but not saved.
- `target_col` *(str, optional)*: If provided, Chokkhu performs target-based analysis (T-Tests, ANOVA, correlations against target). If not provided, it gracefully skips target-based plots without throwing an error.

### 2. Image EDA

Analyze RGB intensity, aspect ratios, blur levels, and class distributions for your image datasets.

```python
import chokkhu as ck

ck.eda.image(
    dataset_path="dataset_folder/", 
    save_dir="reports/image"
)
```

**Parameters:**
- `dataset_path` *(str)*: Path to the root folder of your image dataset (which contains subfolders for each class).
- `save_dir` *(str, optional)*: If provided, all EDA reports are saved to this directory automatically.

---

## Quick Start: Preprocessing

After understanding your data, use Chokkhu's preprocessing modules to prepare it for training.

### 1. Image Preprocessing

Prepare raw image folders for Deep Learning. Chokkhu automatically resizes, normalizes, balances classes using augmentation, and splits the data into stratified Train/Validation/Test arrays.

```python
import chokkhu as ck

# Preprocess, Balance, and Split dataset
train, val, test = ck.preprocessing.image(
    datapath="dataset_folder/"
)

train_X, train_y = train
val_X, val_y = val
test_X, test_y = test
```

### 2. Tabular Preprocessing
*(Feature coming soon)*

```python
import chokkhu as ck

# ck.preprocessing.tabular(dataset_path="dataset.csv") 
```

---

## Training a Deep Learning Model

Once Chokkhu provides your preprocessed `train_X` and `train_y` arrays, you can immediately train a TensorFlow/Keras model without worrying about custom data generators or class imbalances.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train using Chokkhu's preprocessed arrays
model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=10
)
```

---

## Contributing

We welcome contributions! Please check out the [Issues](https://github.com/tamimystic/Chokkhu-PyPi-Package/issues) page if you'd like to help improve Chokkhu.

## License

Distributed under the MIT License. See `LICENSE` for more information.
