import matplotlib.pyplot as plt
import seaborn as sns

from chokkhu.core.visualizer import PlotVisualizer


class ImagePlotter:
    def __init__(self, df, results: dict, save_dir: str, save_reports: bool):
        self.df = df
        self.results = results
        self.save_dir = save_dir
        self.save_reports = save_reports

    def plot_all(self):
        self._plot_structural()
        self._plot_color()
        self._plot_texture()
        self._plot_quality()
        self._plot_deep_learning()

    def _plot_structural(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        class_counts = self.df["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]

        sns.barplot(
            data=class_counts,
            x="Class",
            y="Count",
            hue="Class",
            legend=False,
            palette="viridis",
            ax=axes[0],
        )
        PlotVisualizer.add_bar_labels(axes[0], vertical=True)
        axes[0].set_title("Topic 1: Class-wise Distribution")
        axes[0].tick_params(axis="x", rotation=45)

        sns.histplot(
            data=self.df,
            x="Aspect_Ratio",
            hue="Class",
            kde=False,
            element="step",
            ax=axes[1],
        )
        axes[1].set_title("Topic 1: Aspect Ratio Profiling")

        sns.boxplot(
            data=self.df,
            x="Class",
            y="File_Size_KB",
            hue="Class",
            legend=False,
            palette="Set2",
            ax=axes[2],
        )
        axes[2].set_title("Topic 1: File Storage Size (KB)")

        PlotVisualizer.save_and_show(
            fig, "1_structural_analysis.png", self.save_dir, self.save_reports
        )

    def _plot_color(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        avg_hist = self.results.get("avg_rgb_hist")

        if avg_hist is not None:
            for i, col in enumerate(["red", "green", "blue"]):
                axes[0].plot(avg_hist[:, i], color=col, label=f"{col.upper()}")
                axes[0].fill_between(range(256), avg_hist[:, i], color=col, alpha=0.15)
        axes[0].set_title("Topic 2: Color Intensity Histograms")
        axes[0].legend()

        sns.violinplot(
            data=self.df,
            x="Class",
            y="Brightness",
            hue="Class",
            legend=False,
            palette="coolwarm",
            ax=axes[1],
        )
        axes[1].set_title("Topic 2: Brightness Distribution")

        sns.violinplot(
            data=self.df,
            x="Class",
            y="Contrast",
            hue="Class",
            legend=False,
            palette="coolwarm",
            ax=axes[2],
        )
        axes[2].set_title("Topic 2: Contrast Profiling")

        PlotVisualizer.save_and_show(
            fig, "2_color_analysis.png", self.save_dir, self.save_reports
        )

    def _plot_texture(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="GLCM_Contrast",
            hue="Class",
            legend=False,
            palette="crest",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Topic 3: Texture (GLCM Contrast)")

        sns.boxplot(
            data=self.df,
            x="Class",
            y="Edge_Intensity",
            hue="Class",
            legend=False,
            palette="crest",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Topic 3: Structural Complexity (Edge Density)")

        sns.boxplot(
            data=self.df,
            x="Class",
            y="GLCM_Homogeneity",
            hue="Class",
            legend=False,
            palette="crest",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Topic 3: Texture (GLCM Homogeneity)")

        avg_imgs = self.results.get("avg_images", {})
        keys = list(avg_imgs.keys())
        if keys:
            axes[1, 1].imshow(avg_imgs[keys[0]])
            axes[1, 1].set_title(f"Topic 3: Average Visual (Class: {keys[0]})")
            axes[1, 1].axis("off")

        PlotVisualizer.save_and_show(
            fig, "3_spatial_texture.png", self.save_dir, self.save_reports
        )

    def _plot_quality(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="Shannon_Entropy",
            hue="Class",
            legend=False,
            palette="magma",
            ax=axes[0],
        )
        axes[0].set_title("Topic 4: Shannon Entropy (Information Density)")

        sns.boxplot(
            data=self.df,
            x="Class",
            y="Blur_Score",
            hue="Class",
            legend=False,
            palette="magma",
            ax=axes[1],
        )
        axes[1].set_yscale("log")
        axes[1].set_title("Topic 4: Degradation (Blur/Sharpness)")

        sns.boxplot(
            data=self.df,
            x="Class",
            y="SNR",
            hue="Class",
            legend=False,
            palette="magma",
            ax=axes[2],
        )
        axes[2].set_title("Topic 4: Signal-to-Noise Ratio (SNR)")

        PlotVisualizer.save_and_show(
            fig, "4_quality_analysis.png", self.save_dir, self.save_reports
        )

    def _plot_deep_learning(self):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        sns.scatterplot(
            data=self.df,
            x="PCA1",
            y="PCA2",
            hue="Class",
            palette="tab10",
            alpha=0.7,
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Topic 5: Deep Embedding (PCA)")

        sns.scatterplot(
            data=self.df,
            x="TSNE1",
            y="TSNE2",
            hue="Class",
            palette="tab10",
            alpha=0.7,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Topic 5: Deep Embedding (t-SNE)")

        outliers = self.df["Is_Outlier"].value_counts().reset_index()
        outliers.columns = ["Outlier_Status", "Count"]
        outliers["Outlier_Status"] = outliers["Outlier_Status"].map(
            {-1: "Outlier", 1: "Normal"}
        )
        sns.barplot(
            data=outliers,
            x="Outlier_Status",
            y="Count",
            hue="Outlier_Status",
            legend=False,
            palette="Set1",
            ax=axes[1, 0],
        )
        PlotVisualizer.add_bar_labels(axes[1, 0], vertical=True)
        axes[1, 0].set_title("Topic 5: Anomaly/Outlier Detection")

        dups = self.df["Is_Duplicate"].value_counts().reset_index()
        dups.columns = ["Duplicate_Status", "Count"]
        dups["Duplicate_Status"] = dups["Duplicate_Status"].map(
            {True: "Duplicate", False: "Unique"}
        )
        sns.barplot(
            data=dups,
            x="Duplicate_Status",
            y="Count",
            hue="Duplicate_Status",
            legend=False,
            palette="Set1",
            ax=axes[1, 1],
        )
        PlotVisualizer.add_bar_labels(axes[1, 1], vertical=True)
        axes[1, 1].set_title("Topic 5: Perceptual Duplicate Screening (pHash)")

        PlotVisualizer.save_and_show(
            fig, "5_deep_learning.png", self.save_dir, self.save_reports
        )
