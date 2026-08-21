# Model Training (`ck.train`)

Chokkhu implements classical Machine Learning algorithms **entirely from scratch** (using pure NumPy mathematics). No Scikit-learn dependencies are required. 

The `ck.train()` engine dynamically routes your parameters directly to the underlying model architecture.

## Syntax

```python
import chokkhu as ck

model = ck.train(
    model="random_forest", 
    X_train=X_train, 
    y_train=y_train, 
    task="classification",
    n_estimators=100,      # Dynamic Kwarg
    max_depth=15           # Dynamic Kwarg
)
```

## Core Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | Required | The algorithmic model to use. |
| `X_train` | `np.ndarray` | Required | Training features matrix. |
| `y_train` | `np.ndarray` | `None` | Target labels (Required for Supervised Learning). |
| `task` | `str` | `"auto"` | Resolves to `"classification"` or `"regression"`. |
| `**kwargs` | `Any` | N/A | Hyperparameters specific to the chosen algorithm. |

## Supported Algorithms

### Supervised Learning
*   `linear_regression` (OLS, Ridge, Lasso)
*   `logistic_regression`
*   `knn` (K-Nearest Neighbors)
*   `naive_bayes`
*   `svm` (Support Vector Machine)
*   `decision_tree`
*   `random_forest`
*   `gradient_boosting`

### Unsupervised Learning
*   `kmeans`
*   `dbscan`
*   `hierarchical`

### Reinforcement Learning
*   `q_learning`

??? example "Linear Regression with Regularization (Lasso)"
    ```python
    model = ck.train(
        "linear_regression", 
        X_train, 
        y_train, 
        regularization="lasso", 
        alpha=0.1, 
        epochs=1000, 
        learning_rate=0.01
    )
    ```
