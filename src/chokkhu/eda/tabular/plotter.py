import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chokkhu.core.logger import Logger
from chokkhu.core.visualizer import PlotVisualizer


class TabularPlotter:
    def __init__(
        self,
        df,
        results: dict,
        save_dir: str,
        save_reports: bool,
        target_col: str = None,
    ):
        self.df = df
        self.results = results
        self.save_dir = save_dir
        self.save_reports = save_reports
        self.target_col = target_col

    def plot_all(self):
        Logger.info("Rendering Ultimate Tabular Visualizations...")
        self._plot_metadata()
        self._plot_missing()
        self._plot_numerical()
        self._plot_categorical()
        self._plot_multivariate()
        self._plot_specialized()
        self._plot_advanced()

    def _plot_metadata(self):
        dtypes = self.results.get("metadata", {}).get("dtype_profiling", {})
        if dtypes:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=list(dtypes.values()),
                y=list(dtypes.keys()),
                hue=list(dtypes.keys()),
                legend=False,
                palette="Set2",
                ax=ax,
            )
            ax.set_title("Topic 1: Data Type Profiling")
            PlotVisualizer.save_and_show(
                fig, "1_metadata.png", self.save_dir, self.save_reports
            )

    def _plot_missing(self):
        missing = self.results.get("missing_data", {}).get("missing_density", {})
        if missing:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(missing.values()),
                y=list(missing.keys()),
                hue=list(missing.keys()),
                legend=False,
                palette="Reds_r",
                ax=ax,
            )
            ax.set_title("Topic 2: Missing Value Density (%)")
            PlotVisualizer.save_and_show(
                fig, "2_missing_density.png", self.save_dir, self.save_reports
            )

        null_corr = self.results.get("missing_data", {}).get("null_correlation", None)
        if null_corr is not None and not null_corr.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(null_corr, annot=True, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Topic 2: Nullity Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "2_null_correlation.png", self.save_dir, self.save_reports
            )

    def _plot_numerical(self):
        num_cols = self.results.get("numerical", {}).get("numerical_cols", [])
        if num_cols:
            num_plots = min(len(num_cols), 6)
            fig, axes = plt.subplots(
                int((num_plots + 1) / 2), 2, figsize=(15, 5 * int((num_plots + 1) / 2))
            )
            axes = axes.flatten() if num_plots > 1 else [axes]
            for i, col in enumerate(num_cols[:num_plots]):
                sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], color="teal")
                axes[i].set_title(f"Topic 3: Distribution of {col}")
            for j in range(num_plots, len(axes)):
                axes[j].axis("off")
            PlotVisualizer.save_and_show(
                fig, "3_numerical_distributions.png", self.save_dir, self.save_reports
            )

            fig, axes = plt.subplots(
                int((num_plots + 1) / 2), 2, figsize=(15, 5 * int((num_plots + 1) / 2))
            )
            axes = axes.flatten() if num_plots > 1 else [axes]
            for i, col in enumerate(num_cols[:num_plots]):
                sns.boxplot(x=self.df[col].dropna(), ax=axes[i], color="salmon")
                axes[i].set_title(f"Topic 3: Outliers in {col}")
            for j in range(num_plots, len(axes)):
                axes[j].axis("off")
            PlotVisualizer.save_and_show(
                fig, "3_numerical_outliers.png", self.save_dir, self.save_reports
            )

    def _plot_categorical(self):
        cat_cols = self.results.get("categorical", {}).get("categorical_cols", [])
        if cat_cols:
            cat_plots = min(len(cat_cols), 6)
            fig, axes = plt.subplots(
                int((cat_plots + 1) / 2), 2, figsize=(15, 6 * int((cat_plots + 1) / 2))
            )
            axes = axes.flatten() if cat_plots > 1 else [axes]
            for i, col in enumerate(cat_cols[:cat_plots]):
                # Using vertical bars as requested by user (y=col avoids overlap)
                sns.countplot(
                    data=self.df,
                    y=col,
                    hue=col,
                    legend=False,
                    ax=axes[i],
                    palette="Set2",
                    order=self.df[col].value_counts().index[:15],
                )
                axes[i].set_title(f"Topic 4: Categories in {col}")
            for j in range(cat_plots, len(axes)):
                axes[j].axis("off")
            PlotVisualizer.save_and_show(
                fig, "4_categorical_counts.png", self.save_dir, self.save_reports
            )

    def _plot_multivariate(self):
        pearson = self.results.get("multivariate", {}).get("pearson_corr", None)
        if pearson is not None and not pearson.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                pearson, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, center=0
            )
            ax.set_title("Topic 5: Pearson Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "5_pearson_corr.png", self.save_dir, self.save_reports
            )

        cat_relations = self.results.get("multivariate", {}).get(
            "categorical_relations", []
        )
        if cat_relations:
            df_cr = pd.DataFrame(cat_relations)
            if not df_cr.empty:
                pivot = df_cr.pivot(
                    index="feature_1", columns="feature_2", values="cramers_v"
                ).fillna(0)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax)
                ax.set_title("Topic 5: Cramer's V (Categorical Association)")
                PlotVisualizer.save_and_show(
                    fig, "5_cramers_v.png", self.save_dir, self.save_reports
                )

    def _plot_specialized(self):
        dt_cols = self.results.get("specialized", {}).get("datetime_cols", [])
        if dt_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            col = dt_cols[0]
            series = pd.to_datetime(self.df[col], errors="coerce").dropna()
            series.groupby(series.dt.to_period("M")).size().plot(ax=ax, marker="o")
            ax.set_title(f"Topic 6: Time Series Trend (Monthly Count) for {col}")
            ax.set_ylabel("Count")
            PlotVisualizer.save_and_show(
                fig, "6_datetime_trend.png", self.save_dir, self.save_reports
            )

    def _plot_advanced(self):
        pca_1 = self.results.get("advanced", {}).get("pca_1", None)
        pca_2 = self.results.get("advanced", {}).get("pca_2", None)
        if pca_1 is not None and pca_2 is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            if self.target_col and self.target_col in self.df.columns:
                sns.scatterplot(
                    x=pca_1,
                    y=pca_2,
                    hue=self.df[self.target_col],
                    palette="tab10",
                    ax=ax,
                )
            else:
                sns.scatterplot(x=pca_1, y=pca_2, color="purple", ax=ax)
            ax.set_title("Topic 7: PCA Deep Embedding")
            PlotVisualizer.save_and_show(
                fig, "7_pca_embedding.png", self.save_dir, self.save_reports
            )

        iv_results = self.results.get("advanced", {}).get("information_value", {})
        if iv_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            iv_df = pd.DataFrame(
                list(iv_results.items()), columns=["Feature", "IV"]
            ).sort_values("IV", ascending=False)
            sns.barplot(
                data=iv_df,
                x="IV",
                y="Feature",
                hue="Feature",
                legend=False,
                palette="viridis",
                ax=ax,
            )
            ax.set_title("Topic 7: Information Value (Predictive Power)")
            PlotVisualizer.save_and_show(
                fig, "7_information_value.png", self.save_dir, self.save_reports
            )
