<div align="center">

# 👁️ Chokkhu (চক্ষু)

**The Ultimate Deep Learning & Tabular EDA and Preprocessing Toolkit**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![License](https://img.shields.io/github/license/tamimystic/Chokkhu-PyPi-Package.svg?style=for-the-badge)](https://github.com/tamimystic/Chokkhu-PyPi-Package/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/Chokkhu-PyPi-Package/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/Chokkhu-PyPi-Package/actions)
[![Downloads](https://img.shields.io/pypi/dm/chokkhu.svg?style=for-the-badge)](https://pypi.org/project/chokkhu/)

*Industry-grade data preparation, automated EDA, and class balancing for Computer Vision and Tabular datasets.*

</div>

---

## 📖 Overview

**Chokkhu** (Bengali for "Eye") is a high-performance Python package designed to streamline the most tedious parts of Machine Learning: **Data Analysis and Preprocessing**. 

Whether you are working with large image datasets for Deep Learning (TensorFlow/Keras) or complex Tabular data, Chokkhu automates Exploratory Data Analysis (EDA), missing value handling, class imbalance correction, and generates ultra-professional statistical reports.

### ✨ Key Features
- 📊 **Automated Tabular EDA**: Generates Bivariate, Multivariate, PCA, and VIF visualizations instantly.
- 🖼️ **Comprehensive Image EDA**: Analyzes RGB intensity, aspect ratios, blur levels, and class distributions.
- ⚖️ **Intelligent Class Balancing**: Automatically balances image datasets using advanced augmentation.
- 🚀 **Ready-to-Train Pipelines**: Seamlessly integrates with TensorFlow/Keras (`train_X, train_y`).
- 📈 **Publication-Ready Reports**: Outputs clean, aesthetic charts powered by Seaborn and Matplotlib.

---

## ⚙️ Installation

Install Chokkhu easily via pip. TensorFlow and other scientific libraries are automatically installed as runtime dependencies.

```bash
pip install chokkhu
```

> **Note**: Chokkhu works seamlessly in Google Colab, Jupyter Notebooks, and enterprise CI/CD environments.

---

## 🚀 Quick Start

### 1. Tabular Data Analysis (Auto-EDA)
Generate a massive suite of statistical charts to understand your CSV datasets in one line of code:

```python
import chokkhu as ck

ck.eda.tabular(
    dataset_path="your_dataset.csv",
    save_reports=True,          # Automatically saves all plots as PNGs
    save_dir="tabular_reports", 
    target_col="TargetVariable" # Optional: specify a target for T-Tests & ANOVA
)
```

### 2. Image Dataset EDA & Preprocessing
Prepare raw image folders for Deep Learning. Chokkhu resizes (default 224x224), normalizes, balances classes, and splits the data into Train/Validation/Test sets.

```python
from chokkhu import ImageEDA, ImagePreProcessor

# 1. Analyze your raw image dataset
eda = ImageEDA(dataset_path="dataset/")

# 2. Preprocess, Balance, and Split
processor = ImagePreProcessor(datapath="dataset/")
(train_X, train_y), (val_X, val_y), (test_X, test_y) = processor.get_data()
```

### 3. Training a Deep Learning Model
Once Chokkhu provides your `train_X` and `train_y`, you can immediately train a TensorFlow/Keras model without worrying about custom data generators.

```python
import tensorflow as tf

# Define a custom CNN or use Transfer Learning (e.g., ConvNeXt)
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

## 🤝 Contributing

We welcome contributions! Please check out the [Issues](https://github.com/tamimystic/Chokkhu-PyPi-Package/issues) page if you'd like to help improve Chokkhu.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
