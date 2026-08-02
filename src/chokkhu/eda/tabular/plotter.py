import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chokkhu.core.logger import Logger
from chokkhu.core.visualizer import PlotVisualizer


class TabularPlotter:
    def __init__(
        self,
        df: pd.DataFrame,
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

        # Set clean aesthetic style
        sns.set_theme(style="whitegrid", palette="muted")

    def _add_bar_labels(self, ax, fmt="%.2f"):
        """Adds text labels to the top of vertical bar charts."""
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, padding=3)

    def plot_all(self):
        Logger.info("Rendering Ultimate Statistical Visualizations...")
        self._plot_global()
        self._plot_univariate()
        self._plot_bivariate()
        self._plot_multivariate()

    def _plot_global(self):
        global_res = self.results.get("global_eda", {})

        # Metadata: Data Types
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

        # Missing Data Matrix
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

        # Nullity Correlation
        null_corr = global_res.get("null_correlation", None)
        if null_corr is not None and not null_corr.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                null_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax
            )
            ax.set_title("Nullity Correlation Matrix")
            PlotVisualizer.save_and_show(
                fig, "0_null_correlation.png", self.save_dir, self.save_reports
            )

        # Imputation Shift
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

    def _plot_univariate(self):
        univ = self.results.get("univariate", {})

        # Ordinal
        ordinal = univ.get("ordinal_stats", {})
        for col, stats in ordinal.items():
            freq = stats.get("frequencies", {})
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(freq.keys()),
                y=list(freq.values()),
                hue=list(freq.keys()),
                legend=False,
                palette="pastel",
                ax=ax,
            )
            ax.set_title(f"Frequencies in Ordinal Feature: {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, f"1_ordinal_{col}.png", self.save_dir, self.save_reports
            )

        # Nominal
        nominal = univ.get("nominal_stats", {})
        for col, stats in nominal.items():
            series = self.df[col].dropna()
            top_15 = series.value_counts().head(15)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=top_15.index,
                y=top_15.values,
                hue=top_15.index,
                legend=False,
                palette="Set3",
                ax=ax,
            )
            ax.set_title(f"Top 15 Categories in Nominal Feature: {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=90)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, f"1_nominal_{col}.png", self.save_dir, self.save_reports
            )

        # Discrete
        discrete = univ.get("discrete_stats", {})
        for col, stats in discrete.items():
            freq = stats.get("frequencies", {})
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=list(freq.keys()),
                y=list(freq.values()),
                hue=list(freq.keys()),
                legend=False,
                palette="deep",
                ax=ax,
            )
            ax.set_title(f"Distribution of Discrete Feature: {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            self._add_bar_labels(ax, fmt="%d")
            PlotVisualizer.save_and_show(
                fig, f"2_discrete_{col}.png", self.save_dir, self.save_reports
            )

        # Continuous
        import scipy.stats as sp_stats
        continuous = univ.get("continuous_stats", {})
        for col, stats in continuous.items():
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[0, 0], color="teal")
            axes[0, 0].set_title(f"Distribution of {col}")

            sns.boxplot(y=self.df[col].dropna(), ax=axes[0, 1], color="salmon")
            axes[0, 1].set_title(f"Outliers in {col} (Vertical Boxplot)")

            sns.violinplot(y=self.df[col].dropna(), ax=axes[1, 0], color="skyblue")
            axes[1, 0].set_title(f"Density & Spread in {col} (Violin Plot)")

            sp_stats.probplot(self.df[col].dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f"Q-Q Plot for Normal Distribution of {col}")

            plt.tight_layout()
            PlotVisualizer.save_and_show(
                fig, f"2_continuous_{col}.png", self.save_dir, self.save_reports
            )

        # Specialized: Date-Time
        datetime = univ.get("datetime_stats", {})
        for col, stats in datetime.items():
            fig, ax = plt.subplots(figsize=(12, 6))
            series = pd.to_datetime(self.df[col], errors="coerce").dropna()
            series.groupby(series.dt.to_period("M")).size().plot(
                ax=ax, marker="o", color="navy"
            )
            stat_text = (
                "Stationary" if stats.get("pseudo_stationary") else "Non-Stationary"
            )
            ax.set_title(f"Time Series Trend (Monthly) for {col} [{stat_text}]")
            ax.set_ylabel("Count")
            PlotVisualizer.save_and_show(
                fig, f"3_datetime_{col}.png", self.save_dir, self.save_reports
            )

    def _plot_bivariate(self):
        biv = self.results.get("bivariate", {})

        # Cat vs Cat
        cat_vs_cat = biv.get("cat_vs_cat", {})
        for pair_name, data in cat_vs_cat.items():
            crosstab = data["crosstab"]
            fig, ax = plt.subplots(figsize=(10, 6))
            crosstab.plot(kind='bar', stacked=True, ax=ax, cmap="viridis")
            ax.set_title(f"Categorical vs Categorical: {pair_name}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            PlotVisualizer.save_and_show(fig, f"4_cat_vs_cat_{pair_name}.png", self.save_dir, self.save_reports)

        # Cat vs Num
        cat_vs_num = biv.get("cat_vs_num", {})
        for pair_name, data in cat_vs_num.items():
            cat = data["cat"]
            num = data["num"]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=self.df[cat], y=self.df[num], ax=ax, palette="Set2")
            ax.set_title(f"Categorical vs Numerical: {pair_name} (ANOVA p={data.get('anova_p', 1.0):.4f})")
            ax.tick_params(axis="x", rotation=45)
            PlotVisualizer.save_and_show(fig, f"4_cat_vs_num_{pair_name}.png", self.save_dir, self.save_reports)

        # Num vs Num
        num_vs_num = biv.get("num_vs_num", {})
        for pair_name, data in num_vs_num.items():
            n1 = data["n1"]
            n2 = data["n2"]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=self.df[n1], y=self.df[n2], ax=ax, color="purple")
            ax.set_title(f"Numerical vs Numerical: {pair_name} (Pearson={data.get('pearson', 0.0):.4f})")
            PlotVisualizer.save_and_show(fig, f"4_num_vs_num_{pair_name}.png", self.save_dir, self.save_reports)

        target_analysis = biv.get("target_analysis", {})

        # Information Value (IV)
        iv_res = target_analysis.get("information_value", {})
        if iv_res:
            iv_df = pd.DataFrame(
                list(iv_res.items()), columns=["Feature", "IV"]
            ).sort_values("IV", ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=iv_df,
                x="Feature",
                y="IV",
                hue="Feature",
                legend=False,
                palette="viridis",
                ax=ax,
            )
            ax.set_title(
                f"Predictive Power (Information Value) vs Target: {self.target_col}"
            )
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            PlotVisualizer.save_and_show(
                fig, "4_iv_predictive_power.png", self.save_dir, self.save_reports
            )

        # ANOVA F-Stats equivalent representation (p-values)
        anova = target_analysis.get("categorical_vs_target_anova", {})
        if anova:
            features = list(anova.keys())
            p_vals = [anova[f]["anova_p"] for f in features]
            # Plot -log10(p_value) for better visualization
            log_p = [-np.log10(p + 1e-9) for p in p_vals]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=features, y=log_p, hue=features, legend=False, palette="magma", ax=ax
            )
            ax.set_title(
                f"ANOVA Significance (-log10 P-Value) vs Target: {self.target_col}"
            )
            ax.set_ylabel("-log10(p-value)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            # Add significance line at p=0.05
            ax.axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "4_anova_significance.png", self.save_dir, self.save_reports
            )
            
        # T-Test Representation
        t_tests = target_analysis.get("numerical_vs_target_ttest", {})
        if t_tests:
            features = list(t_tests.keys())
            p_vals = [t_tests[f]["p_val"] for f in features]
            log_p = [-np.log10(p + 1e-9) for p in p_vals]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(
                x=features, y=log_p, hue=features, legend=False, palette="coolwarm", ax=ax
            )
            ax.set_title(
                f"T-Test Significance (-log10 P-Value) vs Target: {self.target_col}"
            )
            ax.set_ylabel("-log10(p-value)")
            ax.tick_params(axis="x", rotation=45)
            self._add_bar_labels(ax)
            ax.axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")
            ax.legend()
            PlotVisualizer.save_and_show(
                fig, "4_ttest_significance.png", self.save_dir, self.save_reports
            )

    def _plot_multivariate(self):
        mult = self.results.get("multivariate", {})

        # Pearson Correlation
        pearson = mult.get("pearson_corr", None)
        if pearson is not None and not pearson.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                pearson, annot=True, cmap="coolwarm", fmt=".2f", center=0, ax=ax
            )
            ax.set_title("Pearson Correlation Matrix (Numerical)")
            PlotVisualizer.save_and_show(
                fig, "5_pearson_correlation.png", self.save_dir, self.save_reports
            )

        # Cramer's V
        cramers = mult.get("cramers_v_matrix", None)
        if cramers is not None and not cramers.empty:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cramers, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
            ax.set_title("Cramer's V Association Matrix (Categorical)")
            PlotVisualizer.save_and_show(
                fig, "5_cramers_v.png", self.save_dir, self.save_reports
            )

        # VIF
        vif = mult.get("vif", {})
        if vif:
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

        # Mutual Information
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

        # Dataset Drift (PSI)
        psi = mult.get("dataset_drift_psi", {})
        if psi:
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

        # PCA
        pca_1 = mult.get("pca_1", None)
        pca_2 = mult.get("pca_2", None)
        pca_index = mult.get("pca_index", None)
        if pca_1 is not None and pca_2 is not None and pca_index is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            if self.target_col and self.target_col in self.df.columns:
                sns.scatterplot(
                    x=pca_1,
                    y=pca_2,
                    hue=self.df.loc[pca_index, self.target_col],
                    palette="tab10",
                    ax=ax,
                )
            else:
                sns.scatterplot(x=pca_1, y=pca_2, color="purple", ax=ax)
            ax.set_title("PCA Deep Embedding (2D Projection)")
            PlotVisualizer.save_and_show(
                fig, "5_pca_embedding.png", self.save_dir, self.save_reports
            )
