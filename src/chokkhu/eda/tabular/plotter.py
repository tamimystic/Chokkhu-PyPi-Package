import matplotlib.pyplot as plt
import seaborn as sns

from chokkhu.core.logger import Logger
from chokkhu.core.visualizer import PlotVisualizer


class TabularPlotter:
    def __init__(self, df, results: dict, save_dir: str, save_reports: bool):
        self.df = df
        self.results = results
        self.save_dir = save_dir
        self.save_reports = save_reports

    def plot_all(self):
        Logger.info("Rendering Tabular Visualizations...")
        self._plot_missing()
        self._plot_correlation()
        self._plot_distributions()
        self._plot_scatter_matrix()
        self._plot_categorical()

    def _plot_missing(self):
        if self.results["missing"].sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                self.df.isnull(), cbar=False, cmap="viridis", yticklabels=False, ax=ax
            )
            ax.set_title("1. Missing Values Heatmap")
            PlotVisualizer.save_and_show(
                fig, "1_missing_values.png", self.save_dir, self.save_reports
            )

    def _plot_correlation(self):
        corr = self.results.get("correlation")
        if corr is not None and not corr.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
            )
            ax.set_title("2. Feature Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "2_correlation_matrix.png", self.save_dir, self.save_reports
            )

    def _plot_distributions(self):
        num_cols = self.results["numerical_cols"]
        if num_cols:
            num_plots = min(len(num_cols), 6)
            fig, axes = plt.subplots(
                int((num_plots + 1) / 2), 2, figsize=(15, 5 * int((num_plots + 1) / 2))
            )
            axes = axes.flatten() if num_plots > 1 else [axes]

            for i, col in enumerate(num_cols[:num_plots]):
                sns.histplot(self.df[col], kde=True, ax=axes[i], color="teal")
                axes[i].set_title(f"3. Distribution of {col}")

            for j in range(num_plots, len(axes)):
                axes[j].axis("off")

            PlotVisualizer.save_and_show(
                fig, "3_numerical_distributions.png", self.save_dir, self.save_reports
            )

    def _plot_scatter_matrix(self):
        num_cols = self.results["numerical_cols"]
        if len(num_cols) > 1:
            cols_to_plot = num_cols[:5]
            pair_grid = sns.pairplot(
                self.df[cols_to_plot].dropna(),
                diag_kind="kde",
                corner=True,
                plot_kws={"alpha": 0.6},
            )
            pair_grid.fig.suptitle("4. Feature Scatter Matrix (Pairplot)", y=1.02)
            PlotVisualizer.save_and_show(
                pair_grid.fig, "4_scatter_matrix.png", self.save_dir, self.save_reports
            )

    def _plot_categorical(self):
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
                PlotVisualizer.add_bar_labels(axes[i], vertical=False)
                axes[i].set_title(f"5. Top Categories in {col}")

            for j in range(cat_plots, len(axes)):
                axes[j].axis("off")

            PlotVisualizer.save_and_show(
                fig, "5_categorical_counts.png", self.save_dir, self.save_reports
            )
