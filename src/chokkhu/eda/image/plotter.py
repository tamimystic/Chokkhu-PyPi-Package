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

    def _save_and_close(self, fig, filename):
        PlotVisualizer.save_and_show(fig, filename, self.save_dir, self.save_reports)
        plt.close(fig)

    def _plot_structural(self):
        # 1. Class Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = self.df["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]

        sns.barplot(
            data=class_counts,
            x="Class",
            y="Count",
            hue="Class",
            legend=False,
            palette="viridis",
            ax=ax,
        )
        PlotVisualizer.add_bar_labels(ax, vertical=True)
        ax.set_title("Class-wise Distribution")
        ax.tick_params(axis="x", rotation=45)
        self._save_and_close(fig, "1_class_distribution.png")

        # 2. Aspect Ratio Profiling
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=self.df,
            x="Aspect_Ratio",
            hue="Class",
            kde=False,
            element="step",
            ax=ax,
        )
        ax.set_title("Aspect Ratio Profiling")
        self._save_and_close(fig, "1_aspect_ratio.png")

        # 3. File Storage Size
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="File_Size_KB",
            hue="Class",
            legend=False,
            palette="Set2",
            ax=ax,
        )
        ax.set_title("File Storage Size (KB)")
        self._save_and_close(fig, "1_file_size.png")

    def _plot_color(self):
        # 1. Color Intensity
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_hist = self.results.get("avg_rgb_hist")

        if avg_hist is not None:
            for i, col in enumerate(["red", "green", "blue"]):
                ax.plot(avg_hist[:, i], color=col, label=f"{col.upper()}")
                ax.fill_between(range(256), avg_hist[:, i], color=col, alpha=0.15)
        ax.set_title("Color Intensity Histograms")
        ax.legend()
        self._save_and_close(fig, "2_color_intensity.png")

        # 2. Brightness
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=self.df,
            x="Class",
            y="Brightness",
            hue="Class",
            legend=False,
            palette="coolwarm",
            ax=ax,
        )
        ax.set_title("Brightness Distribution")
        self._save_and_close(fig, "2_brightness.png")

        # 3. Contrast
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(
            data=self.df,
            x="Class",
            y="Contrast",
            hue="Class",
            legend=False,
            palette="coolwarm",
            ax=ax,
        )
        ax.set_title("Contrast Profiling")
        self._save_and_close(fig, "2_contrast.png")

    def _plot_texture(self):
        # 1. GLCM Contrast
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="GLCM_Contrast",
            hue="Class",
            legend=False,
            palette="crest",
            ax=ax,
        )
        ax.set_title("Texture (GLCM Contrast)")
        self._save_and_close(fig, "3_texture_contrast.png")

        # 2. Edge Density
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="Edge_Intensity",
            hue="Class",
            legend=False,
            palette="crest",
            ax=ax,
        )
        ax.set_title("Structural Complexity (Edge Density)")
        self._save_and_close(fig, "3_edge_density.png")

        # 3. GLCM Homogeneity
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="GLCM_Homogeneity",
            hue="Class",
            legend=False,
            palette="crest",
            ax=ax,
        )
        ax.set_title("Texture (GLCM Homogeneity)")
        self._save_and_close(fig, "3_texture_homogeneity.png")

        # 4. Average Images
        avg_imgs = self.results.get("avg_images", {})
        for class_name, img in avg_imgs.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(f"Average Visual (Class: {class_name})")
            ax.axis("off")
            self._save_and_close(fig, f"3_avg_image_{class_name}.png")

    def _plot_quality(self):
        # 1. Shannon Entropy
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="Shannon_Entropy",
            hue="Class",
            legend=False,
            palette="magma",
            ax=ax,
        )
        ax.set_title("Shannon Entropy (Information Density)")
        self._save_and_close(fig, "4_entropy.png")

        # 2. Blur Score
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="Blur_Score",
            hue="Class",
            legend=False,
            palette="magma",
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title("Degradation (Blur/Sharpness)")
        self._save_and_close(fig, "4_blur.png")

        # 3. SNR
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=self.df,
            x="Class",
            y="SNR",
            hue="Class",
            legend=False,
            palette="magma",
            ax=ax,
        )
        ax.set_title("Signal-to-Noise Ratio (SNR)")
        self._save_and_close(fig, "4_snr.png")
