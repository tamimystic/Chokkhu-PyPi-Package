from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chokkhu.core.visualizer import PlotVisualizer

from .base_plotter import BasePlotter


class MultivariatePlotter(BasePlotter):

    def plot(self):
        mult = self.results.get("multivariate", {})
        pearson = mult.get("pearson_corr", None)
        if pearson is not None and (not pearson.empty):
            print("  Correlation Analysis (Pearson, Spearman)")
            g = sns.clustermap(
                pearson,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                center=0,
                figsize=(12, 10),
            )
            g.fig.suptitle(
                "Pearson Correlation Matrix (Hierarchical Clustering)", y=1.05
            )
            PlotVisualizer.save_and_show(
                g.fig, "5_pearson_correlation.png", self.save_dir, self.save_reports
            )
        spearman = mult.get("spearman_corr", None)
        if spearman is not None and (not spearman.empty):
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(spearman, annot=True, cmap="mako", fmt=".2f", center=0, ax=ax)
            ax.set_title("Spearman Correlation Matrix (Non-Linear Monotonic)")
            PlotVisualizer.save_and_show(
                fig, "5_spearman_correlation.png", self.save_dir, self.save_reports
            )
        cramers = mult.get("cramers_v_matrix", None)
        if cramers is not None and (not cramers.empty):
            print("  Association Analysis (Cramer's V, Mutual Information)")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cramers, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_title("Cramer's V Association Matrix (Categorical)")
            PlotVisualizer.save_and_show(
                fig, "5_cramers_v.png", self.save_dir, self.save_reports
            )
        vif = mult.get("vif", {})
        if vif:
            print("  Multicollinearity Analysis (VIF)")
            vif_df = pd.DataFrame(
                list(vif.items()), columns=["Feature", "VIF"]
            ).sort_values("VIF", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=vif_df,
                x="Feature",
                y="VIF",
                hue="Feature",
                legend=False,
                palette="Reds",
                ax=ax,
            )
            ax.set_title("Variance Inflation Factor (Multicollinearity)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            ax.axhline(
                y=5.0, color="r", linestyle="--", label="High VIF Threshold (5.0)"
            )
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "5_vif.png", self.save_dir, self.save_reports
            )
        mi = mult.get("mutual_information", {})
        if mi:
            mi_df = pd.DataFrame(
                list(mi.items()), columns=["Feature", "MI"]
            ).sort_values("MI", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=mi_df,
                x="Feature",
                y="MI",
                hue="Feature",
                legend=False,
                palette="plasma",
                ax=ax,
            )
            ax.set_title(f"Mutual Information vs Target: {self.target_col}")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            PlotVisualizer.save_and_show(
                fig, "5_mutual_information.png", self.save_dir, self.save_reports
            )
        psi = mult.get("dataset_drift_psi", {})
        if psi:
            print("  Data Drift & Stability (PSI)")
            psi_df = pd.DataFrame(
                list(psi.items()), columns=["Feature", "PSI"]
            ).sort_values("PSI", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=psi_df,
                x="Feature",
                y="PSI",
                hue="Feature",
                legend=False,
                palette="Oranges",
                ax=ax,
            )
            ax.set_title("Dataset Drift (Population Stability Index over Random Split)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            ax.axhline(y=0.2, color="r", linestyle="--", label="Drift Threshold (0.2)")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "5_dataset_drift.png", self.save_dir, self.save_reports
            )
        pca_1 = mult.get("pca_1", None)
        pca_2 = mult.get("pca_2", None)
        pca_index = mult.get("pca_index", None)
        if pca_1 is not None and pca_2 is not None and (pca_index is not None):
            print("  Dimensionality Reduction (PCA)")
            fig, ax = plt.subplots(figsize=(10, 8))
            if self.target_col and self.target_col in self.df.columns:
                target_data = self.df.loc[pca_index, self.target_col]
                if (
                    pd.api.types.is_numeric_dtype(target_data)
                    and target_data.nunique() > 15
                ):
                    sns.scatterplot(
                        x=pca_1,
                        y=pca_2,
                        hue=target_data,
                        palette="viridis",
                        legend=False,
                        ax=ax,
                    )
                    norm = plt.Normalize(target_data.min(), target_data.max())
                    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax)
                    cbar.set_label(self.target_col)
                else:
                    sns.scatterplot(
                        x=pca_1, y=pca_2, hue=target_data, palette="tab10", ax=ax
                    )
            else:
                sns.scatterplot(x=pca_1, y=pca_2, color="purple", ax=ax)
            ax.set_title("PCA Deep Embedding (2D Projection)")
            PlotVisualizer.save_and_show(
                fig, "5_pca_embedding.png", self.save_dir, self.save_reports
            )
        mahal = mult.get("mahalanobis_outliers", {})
        if mahal and mahal.get("count", 0) > 0:
            print(
                f"  Multivariate Outliers Detected: {mahal.get('count')} (Mahalanobis)"
            )
            distances = mahal.get("distances", [])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(distances, kde=True, color="crimson", ax=ax, bins=30)
            ax.set_title(
                f"Mahalanobis Distance Distribution ({mahal.get('count')} Outliers Detected)"
            )
            ax.set_xlabel("Mahalanobis Distance")
            ax.set_ylabel("Frequency")
            PlotVisualizer.save_and_show(
                fig, "5_mahalanobis_outliers.png", self.save_dir, self.save_reports
            )
