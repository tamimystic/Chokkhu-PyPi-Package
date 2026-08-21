# Welcome to Chokkhu 👁️

[![PyPI version](https://badge.fury.io/py/chokkhu.svg)](https://badge.fury.io/py/chokkhu)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Minimalistic Code. Maximum Output. Zero Heavy Dependencies.**

Chokkhu is a high-performance Python toolkit designed to streamline the complete Machine Learning lifecycle. From raw data loading to exploratory data analysis (EDA), cleaning, preprocessing, feature transformation, model training (from scratch), and evaluation—all accessible via a beautiful, unified, one-line API.

---

## 🎯 Why Chokkhu?

- **Zero Heavy Dependencies**: Built entirely on NumPy, Pandas, SciPy, Matplotlib, Seaborn, and OpenCV-headless. No PyTorch, TensorFlow, or Scikit-Learn bloat!
- **Unified Minimalist API**: Powerful operations encapsulated in intuitive functions (ck.load, ck.eda, ck.clean, ck.preprocess, ck.transform, ck.split, ck.train, ck.evaluate).
- **End-to-End Pipeline**: Handle Tabular and Image data pipelines seamlessly without leaving your notebook.
- **Beautiful HTML Reports**: Interactive, publication-ready EDA reports generated instantly.

---

## 🚀 Quick Glance

`python
import chokkhu as ck

# 1. Load Data
df = ck.load("dataset.csv")

# 2. Automated EDA
ck.eda.tabular("dataset.csv", target_col="price", save_reports=True)

# 3. Clean
df_clean = ck.clean(df, missing="knn", outliers="isolation_forest", duplicates=True)

# 4. Preprocess
df_proc, state = ck.preprocess(df_clean, target="price", scale="standard", encode="onehot")

# 5. Split
X_train, X_test, y_train, y_test = ck.split(df_proc, target="price")

# 6. Train a Model (Implemented from Scratch!)
model = ck.train("random_forest", X_train, y_train)

# 7. Evaluate
results = ck.evaluate(model, X_test, y_test)
`

Ready to dive in? Head over to the [Getting Started](getting_started.md) guide!
