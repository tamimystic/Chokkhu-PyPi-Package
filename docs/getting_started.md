# Getting Started

## Installation

Install Chokkhu directly from the Python Package Index (PyPI) via pip:

`ash
pip install --upgrade chokkhu
`

## Core Requirements

Chokkhu is designed to be extremely lightweight. Its core dependencies are automatically installed during the pip installation process:
- 
umpy
- pandas
- scipy
- matplotlib
- seaborn
- opencv-python-headless
- 	qdm

## Functional API Paradigm

Chokkhu operates on a strict functional API paradigm. Rather than instantiating complex class objects for basic tasks, you pass your dataset through a series of pure functions. 

Each function utilizes **kwargs to allow maximum customizability for advanced users, while maintaining sensible, robust defaults for beginners.

### Example Workflow

`python
import chokkhu as ck

# Load the dataset
df = ck.load("dataset.csv")

# Perform automated EDA
ck.eda.tabular(df, target_col="target")

# Clean, Preprocess, and Split
df_clean = ck.clean(df, missing="knn", outliers="isolation_forest")
df_proc, state = ck.preprocess(df_clean, target="target", scale="standard")
X_train, X_test, y_train, y_test = ck.split(df_proc, target="target")

# Train and Evaluate
model = ck.train("random_forest", X_train, y_train, n_estimators=100)
results = ck.evaluate(model, X_test, y_test)
`

Please navigate to the **API Reference** section to explore the detailed parameters for each function.
