from __future__ import annotations
import numpy as np
from chokkhu.core.logger import Logger
from chokkhu.evaluation.metrics import (
    accuracy_score,
    precision_recall_f1,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from chokkhu.core.visualizer import PlotVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate(
    model,
    X_test,
    y_test,
    task: str = "auto",
    average: str = "macro",
    save_reports: bool = False,
    save_dir: str = "chokkhu_reports",
) -> dict:
    X = np.asarray(X_test) if not isinstance(X_test, np.ndarray) else X_test
    y = np.asarray(y_test) if not isinstance(y_test, np.ndarray) else y_test

    if hasattr(X_test, "values"):
        X = X_test.values
    if hasattr(y_test, "values"):
        y = y_test.values

    y = y.flatten()

    Logger.info(f"Generating predictions for {len(y)} samples...")
    y_pred = model.predict(X)

    if task == "auto":
        if hasattr(model, "task"):
            task = model.task
        else:
            task = "classification" if len(np.unique(y)) <= 20 else "regression"

    results = {}

    if save_reports:
        os.makedirs(save_dir, exist_ok=True)

    if task == "classification":
        acc = accuracy_score(y, y_pred)
        prec, rec, f1 = precision_recall_f1(y, y_pred, average=average)
        cm, classes = confusion_matrix(y, y_pred)

        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
        }

        md_text = (
            "## Model Evaluation (Classification)\n"
            f"- **Accuracy**: {acc:.4f}\n"
            f"- **Precision ({average})**: {prec:.4f}\n"
            f"- **Recall ({average})**: {rec:.4f}\n"
            f"- **F1-Score ({average})**: {f1:.4f}\n"
        )
        PlotVisualizer.display_markdown(md_text)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        PlotVisualizer.save_and_show(
            fig, "confusion_matrix.png", save_dir, save_reports
        )

    elif task == "regression":
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results = {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}

        md_text = (
            "## Model Evaluation (Regression)\n"
            f"- **MSE**: {mse:.4f}\n"
            f"- **RMSE**: {rmse:.4f}\n"
            f"- **MAE**: {mae:.4f}\n"
            f"- **R2-Score**: {r2:.4f}\n"
        )
        PlotVisualizer.display_markdown(md_text)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y, y_pred, alpha=0.5, color="b")
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--")
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        PlotVisualizer.save_and_show(
            fig, "actual_vs_predicted.png", save_dir, save_reports
        )

    else:
        raise ValueError(
            f"Unknown task type: {task}. Use 'classification' or 'regression'."
        )

    Logger.info("Model evaluation completed successfully.")
    return results
