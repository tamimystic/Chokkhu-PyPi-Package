# End-to-End Tabular Pipeline

Below is a complete, real-world example of using Chokkhu to build an end-to-end Machine Learning pipeline on tabular data.

```python
import chokkhu as ck

# 1. Load Data
df = ck.load("dataset.csv")

# 2. EDA
ck.eda.tabular("dataset.csv", target_col="price", save_reports=True)

# 3. Clean
df_clean = ck.clean(
    df,
    missing="knn",
    outliers="isolation_forest",
    duplicates=True,
    fix_data_types=True
)

# 4. Preprocess
df_proc, state = ck.preprocess(
    df_clean,
    target="price",
    scale="standard",
    encode="onehot"
)

# 5. Transform (Dimensionality Reduction)
df_trans = ck.transform(
    df_proc,
    target="price",
    pca=10
)

# 6. Split
X_train, X_test, y_train, y_test = ck.split(
    df_trans,
    target="price",
    test_size=0.2,
    stratify=True,
    random_state=42
)

# 7. Train Model (Random Forest from scratch!)
model = ck.train(
    "random_forest", 
    X_train.values if hasattr(X_train, 'values') else X_train, 
    y_train.values if hasattr(y_train, 'values') else y_train, 
    n_estimators=100, 
    max_depth=10
)

# 8. Evaluate
results = ck.evaluate(model, X_test.values if hasattr(X_test, 'values') else X_test, y_test.values if hasattr(y_test, 'values') else y_test, save_reports=True)
print(results)
```
