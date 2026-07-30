import os
import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .visualizer import PlotVisualizer

warnings.filterwarnings("ignore", category=FutureWarning)


class TabularEDA:
    def __init__(
        self,
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports_Tabular",
    ):
        """
        Initializes the TabularEDA class and triggers the analysis pipeline for CSV/Excel data.
        """
        self.dataset_path: str = dataset_path
        self.save_reports: bool = save_reports
        self.save_dir: str = save_dir
        self.df: pd.DataFrame = pd.DataFrame()
        self.results: Dict[str, Any] = {}

        if self.save_reports:
            os.makedirs(self.save_dir, exist_ok=True)

        PlotVisualizer.setup_theme()
        self._perform_eda()

    def _perform_eda(self) -> None:
        print(f"--- Executing Tabular EDA for: {self.dataset_path} ---")
        self._load_data()
        if self.df.empty:
            print("Error: Could not load data or dataset is empty.")
            return

        self._analyze_data()
        self._visual_reports()

    def _load_data(self) -> None:
        try:
            if self.dataset_path.endswith(".csv"):
                self.df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith((".xls", ".xlsx")):
                self.df = pd.read_excel(self.dataset_path)
            else:
                print("Unsupported file format. Please provide a CSV or Excel file.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def _analyze_data(self) -> None:
        self.results["shape"] = self.df.shape
        self.results["dtypes"] = self.df.dtypes
        self.results["missing"] = self.df.isnull().sum()
        self.results["numerical_cols"] = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.results["categorical_cols"] = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if len(self.results["numerical_cols"]) > 0:
            self.results["correlation"] = self.df[self.results["numerical_cols"]].corr()

    def _visual_reports(self) -> None:
        # 1. Missing Values Heatmap
        if self.results["missing"].sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                self.df.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax
            )
            ax.set_title("Missing Values Heatmap")
            PlotVisualizer.save_and_show(
                fig, "1_missing_values.png", self.save_dir, self.save_reports
            )

        # 2. Correlation Matrix
        if "correlation" in self.results and not self.results["correlation"].empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                self.results["correlation"],
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Feature Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "2_correlation_matrix.png", self.save_dir, self.save_reports
            )

        # 3. Numerical Distributions
        num_cols = self.results["numerical_cols"]
        if num_cols:
            num_plots = min(
                len(num_cols), 6
            )  # Plot up to 6 numerical columns to avoid clutter
            fig, axes = plt.subplots(
                int((num_plots + 1) / 2), 2, figsize=(15, 5 * int((num_plots + 1) / 2))
            )
            axes = axes.flatten() if num_plots > 1 else [axes]

            for i, col in enumerate(num_cols[:num_plots]):
                sns.histplot(self.df[col], kde=True, ax=axes[i], color="teal")
                axes[i].set_title(f"Distribution of {col}")

            # Hide empty subplots
            for j in range(num_plots, len(axes)):
                axes[j].axis("off")

            PlotVisualizer.save_and_show(
                fig, "3_numerical_distributions.png", self.save_dir, self.save_reports
            )

        # 4. Categorical Counts
        cat_cols = self.results["categorical_cols"]
        if cat_cols:
            cat_plots = min(len(cat_cols), 4)
            fig, axes = plt.subplots(
                int((cat_plots + 1) / 2), 2, figsize=(15, 6 * int((cat_plots + 1) / 2))
            )
            axes = axes.flatten() if cat_plots > 1 else [axes]

            for i, col in enumerate(cat_cols[:cat_plots]):
                sns.countplot(
                    data=self.df,
                    y=col,
                    ax=axes[i],
                    palette="Set2",
                    order=self.df[col].value_counts().index[:10],
                )
                axes[i].set_title(f"Top Categories in {col}")

            for j in range(cat_plots, len(axes)):
                axes[j].axis("off")

            PlotVisualizer.save_and_show(
                fig, "4_categorical_counts.png", self.save_dir, self.save_reports
            )

        if self.save_reports:
            print(f"\n[INFO] Tabular reports saved in: {self.save_dir}")
