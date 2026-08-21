# Model Evaluation (`ck.evaluate`)

Evaluate your trained models instantly with comprehensive mathematical metrics and high-quality visual heatmaps.

## Syntax

```python
import chokkhu as ck

results = ck.evaluate(
    model=model, 
    X_test=X_test, 
    y_test=y_test, 
    save_reports=True, 
    save_dir="./eval_reports"
)
print(results)
```

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `Object` | Required | The trained Chokkhu model instance. |
| `X_test` | `np.ndarray` | Required | Testing features matrix. |
| `y_test` | `np.ndarray` | Required | True target labels. |
| `task` | `str` | `"auto"` | Task resolution: `"classification"` or `"regression"`. |
| `average` | `str` | `"macro"` | Averaging method for multi-class metrics (Options: `"macro"`, `"weighted"`). |
| `save_reports` | `bool` | `False` | If True, saves evaluation visualizations to disk. |
| `save_dir` | `str` | `"chokkhu_reports"` | Target directory for saved visualizations. |

## Internal Metrics Generated

### Classification Metrics
- **Accuracy**
- **Precision** (Macro/Weighted)
- **Recall** (Macro/Weighted)
- **F1-Score** (Macro/Weighted)
- **Confusion Matrix** (Raw array and Seaborn Heatmap rendering)

### Regression Metrics
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R2-Score** (Coefficient of Determination)
- **Actual vs Predicted Plot**
