from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chokkhu.core.visualizer import PlotVisualizer

from .base_plotter import BasePlotter


class GlobalPlotter(BasePlotter):

    def plot(self):
        global_res = self.results.get("global_eda", {})
        dtypes = global_res.get("dtype_profiling", {})
        if dtypes:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(dtypes.keys()),
                y=list(dtypes.values()),
                hue=list(dtypes.keys()),
                legend=False,
                palette="Set2",
                ax=ax,
            )
            ax.set_title("Data Type Distribution")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, "0_global_dtypes.png", self.save_dir, self.save_reports
            )
        missing_matrix = global_res.get("missing_matrix", pd.DataFrame())
        if not missing_matrix.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                missing_matrix, cmap="binary", cbar=False, yticklabels=False, ax=ax
            )
            ax.set_title("Missing Value Matrix (White = Valid, Black = Missing)")
            PlotVisualizer.save_and_show(
                fig, "0_missing_matrix.png", self.save_dir, self.save_reports
            )
        null_corr = global_res.get("null_correlation", None)
        if null_corr is not None and (not null_corr.empty):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                null_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax
            )
            ax.set_title("Nullity Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "0_null_correlation.png", self.save_dir, self.save_reports
            )
        imputation_shift = global_res.get("imputation_shift", {})
        if imputation_shift:
            features = list(imputation_shift.keys())
            shifts = [v["shift_pct_mean"] for v in imputation_shift.values()]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=features, y=shifts, hue=features, legend=False, palette="Reds", ax=ax
            )
            ax.set_title("Variance Shift (%) if Imputed with Mean")
            ax.set_ylabel("Shift (%)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            PlotVisualizer.save_and_show(
                fig, "0_imputation_shift.png", self.save_dir, self.save_reports
            )
