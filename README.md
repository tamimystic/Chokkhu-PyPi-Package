<div align="center">

<img src="https://raw.githubusercontent.com/tamimystic/Chokkhu-PyPi-Package/main/profile.jpg" width="150" height="150" style="border-radius:50%;" alt="Author Profile">

# Chokkhu

**A Professional Statistical EDA Toolkit for Computer Vision & Tabular Datasets**

[![PyPI version](https://img.shields.io/pypi/v/chokkhu.svg?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/chokkhu/)
[![Python versions](https://img.shields.io/pypi/pyversions/chokkhu.svg?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/chokkhu/)
[![License](https://img.shields.io/github/license/tamimystic/Chokkhu-PyPi-Package.svg?style=for-the-badge)](https://github.com/tamimystic/Chokkhu-PyPi-Package/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tamimystic/Chokkhu-PyPi-Package/ci.yml?branch=main&style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/tamimystic/Chokkhu-PyPi-Package/actions)
[![Format](https://img.shields.io/pypi/format/chokkhu?style=for-the-badge)](https://pypi.org/project/chokkhu/)
[![Status](https://img.shields.io/pypi/status/chokkhu?style=for-the-badge)](https://pypi.org/project/chokkhu/)

*Industry-grade data analysis and visual reports generation.*

</div>

---

## Overview

**Chokkhu** is a high-performance Python package designed to streamline Exploratory Data Analysis (EDA). 

Designed to be incredibly lightweight without relying on massive machine learning frameworks (No TensorFlow, PyTorch, or Scikit-Learn dependencies), it perfectly generates unified, modular statistical reports.

---

## Installation

Install Chokkhu via pip.

```bash
pip install chokkhu
```

---

## Quick Start: Exploratory Data Analysis (EDA)

Chokkhu provides a unified, clean API for generating EDA reports.

### 1. Tabular EDA

Generate a comprehensive suite of statistical charts (Univariate, Bivariate, Multivariate, Outliers) to understand your CSV datasets in one simple call:

```python
import chokkhu as ck

csv_dataset_path = '/kaggle/input/datasets/laveshjadon/ai-impact-on-students/ai_student_impact_dataset (1).csv'

tabular_eda = ck.eda.tabular(
    dataset_path=csv_dataset_path, 
    save_reports=True, 
    save_dir='/kaggle/working/tabular_reports',
    target_col='Pre_Semester_GPA'
)
```

**Parameters:**
- `dataset_path` *(str)*: Path to your CSV file.
- `save_reports` *(bool)*: Whether to save the plots to the disk.
- `save_dir` *(str)*: Directory where reports will be saved automatically.
- `target_col` *(str, optional)*: If provided, Chokkhu performs target-based analysis.

---

### 2. Image EDA

Analyze image data quality, structural integrity, spatial textures, and color profiles for your datasets. Each topic metric is plotted cleanly and saved directly.

```python
import chokkhu as ck

image_dataset_path = '/kaggle/input/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset'

image_eda = ck.eda.image(
    dataset_path=image_dataset_path, 
    save_reports=True, 
    save_dir='/kaggle/working/image_reports'
)
```

**Parameters:**
- `dataset_path` *(str)*: Path to the root folder of your image dataset.
- `save_reports` *(bool)*: Whether to save the plots to the disk.
- `save_dir` *(str)*: Directory where all EDA reports will be saved automatically.

---

## Contributing

We welcome contributions! Please check out the [Issues](https://github.com/tamimystic/Chokkhu-PyPi-Package/issues) page if you'd like to help improve Chokkhu.

## License

Distributed under the MIT License. See `LICENSE` for more information.
