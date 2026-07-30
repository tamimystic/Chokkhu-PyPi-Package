import os
import warnings
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from .visualizer import PlotVisualizer

warnings.filterwarnings("ignore", category=FutureWarning)


class ImageEDA:
    def __init__(
        self,
        dataset_path: str,
        save_reports: bool = True,
        save_dir: str = "chokkhu_outputs/EDA_Reports",
    ):
        """
        Initializes the Ultra Pro Max ImageEDA class and triggers the analysis pipeline.
        """
        self.dataset_path: str = dataset_path
        self.save_reports: bool = save_reports
        self.save_dir: str = save_dir
        self.results: Dict[str, Any] = {}
        self.class_paths: List[str] = []

        if self.save_reports:
            os.makedirs(self.save_dir, exist_ok=True)

        PlotVisualizer.setup_theme()
        self._perform_eda()

    def _perform_eda(self) -> None:
        print(f"--- Executing Ultra Pro Max EDA for: {self.dataset_path} ---")
        self._collect_paths()
        if not self.class_paths:
            print("Error: No valid images found in the specified path.")
            return
        self.results = self._analyze_data()
        self._visual_reports()

    def _collect_paths(self) -> None:
        for root, _, files in os.walk(self.dataset_path):
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                self.class_paths.append(root)

    def _analyze_data(self) -> Dict[str, Any]:
        exts = (".png", ".jpg", ".jpeg")
        metrics_list = []
        total_rgb_hist = np.zeros((256, 3))
        processed_count = 0
        pca_samples = []
        pca_labels = []

        # Max samples for PCA per class to avoid memory overload
        MAX_PCA_SAMPLES = 200

        for path in self.class_paths:
            class_name = os.path.basename(path)
            files = [f for f in os.listdir(path) if f.lower().endswith(exts)]

            pca_sampled = 0
            for img_name in tqdm(files, desc=f"Processing {class_name}"):
                img_path = os.path.join(path, img_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                h, w, _ = img_rgb.shape

                # Metrics Extraction
                brightness = gray.mean()
                contrast = gray.std()
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                edges = cv2.Canny(gray, 100, 200).mean()

                metrics_list.append(
                    {
                        "Class": class_name,
                        "Image": img_name,
                        "Width": w,
                        "Height": h,
                        "Aspect_Ratio": w / h if h > 0 else 0,
                        "Brightness": brightness,
                        "Contrast": contrast,
                        "Blur_Score": blur,
                        "Edge_Intensity": edges,
                    }
                )

                # RGB Distribution
                for i in range(3):
                    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                    total_rgb_hist[:, i] += hist.flatten()  # type: ignore

                # Sampling for PCA (Downscale to 32x32 for memory efficiency)
                if pca_sampled < MAX_PCA_SAMPLES:
                    resized_for_pca = cv2.resize(gray, (32, 32)).flatten()
                    pca_samples.append(resized_for_pca)
                    pca_labels.append(class_name)
                    pca_sampled += 1

                processed_count += 1

        df_metrics = pd.DataFrame(metrics_list)
        avg_hist = (
            total_rgb_hist / processed_count if processed_count > 0 else total_rgb_hist
        )

        # PCA Computation
        df_pca = None
        if pca_samples:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(np.array(pca_samples))
            df_pca = pd.DataFrame(
                {
                    "PCA1": pca_result[:, 0],
                    "PCA2": pca_result[:, 1],
                    "Class": pca_labels,
                }
            )

        return {
            "df_metrics": df_metrics,
            "avg_rgb_hist": avg_hist,
            "df_pca": df_pca,
            "total_classes": len(self.class_paths),
            "total_images": processed_count,
        }

    def _visual_reports(self) -> None:
        df = self.results["df_metrics"]

        # 1. Class Distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        class_counts = df["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        sns.barplot(data=class_counts, x="Class", y="Count", palette="viridis", ax=ax)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",  # type: ignore
                (p.get_x() + p.get_width() / 2.0, p.get_height()),  # type: ignore
                ha="center",
                va="center",
                xytext=(0, 8),
                textcoords="offset points",
            )
        ax.set_title("Class-wise Image Distribution")
        ax.tick_params(axis="x", rotation=45)
        PlotVisualizer.save_and_show(
            fig, "1_class_distribution.png", self.save_dir, self.save_reports
        )

        # 2. Dimensions & Aspect Ratio
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(
            data=df,
            x="Width",
            hue="Class",
            kde=False,
            element="step",
            ax=axes[0],
            palette="Set2",
        )
        axes[0].set_title("Width Distribution by Class")
        sns.histplot(
            data=df,
            x="Height",
            hue="Class",
            kde=False,
            element="step",
            ax=axes[1],
            palette="Set2",
        )
        axes[1].set_title("Height Distribution by Class")
        sns.boxplot(data=df, x="Class", y="Aspect_Ratio", ax=axes[2], palette="Set2")
        axes[2].set_title("Aspect Ratio by Class")
        axes[2].tick_params(axis="x", rotation=45)
        PlotVisualizer.save_and_show(
            fig, "2_dimension_analysis.png", self.save_dir, self.save_reports
        )

        # 3. Quality Metrics (Brightness, Contrast, Blur, Edges)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.violinplot(
            data=df, x="Class", y="Brightness", palette="coolwarm", ax=axes[0, 0]
        )
        axes[0, 0].set_title("Brightness Distribution")
        axes[0, 0].tick_params(axis="x", rotation=45)

        sns.violinplot(
            data=df, x="Class", y="Contrast", palette="coolwarm", ax=axes[0, 1]
        )
        axes[0, 1].set_title("Contrast Distribution")
        axes[0, 1].tick_params(axis="x", rotation=45)

        sns.boxplot(data=df, x="Class", y="Blur_Score", palette="crest", ax=axes[1, 0])
        axes[1, 0].set_title("Blur Score (Laplacian Variance)")
        axes[1, 0].set_yscale("log")
        axes[1, 0].tick_params(axis="x", rotation=45)

        sns.boxplot(
            data=df, x="Class", y="Edge_Intensity", palette="crest", ax=axes[1, 1]
        )
        axes[1, 1].set_title("Edge Intensity (Canny)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        PlotVisualizer.save_and_show(
            fig, "3_quality_metrics.png", self.save_dir, self.save_reports
        )

        # 4. RGB Intensity Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, col in enumerate(["red", "green", "blue"]):
            ax.plot(
                self.results["avg_rgb_hist"][:, i],
                color=col,
                label=f"{col.upper()} Channel",
                linewidth=2,
            )
            ax.fill_between(
                range(256), self.results["avg_rgb_hist"][:, i], color=col, alpha=0.15
            )
        ax.set_title("Global Average RGB Intensity Distribution")
        ax.legend()
        PlotVisualizer.save_and_show(
            fig, "4_rgb_intensity.png", self.save_dir, self.save_reports
        )

        # 5. PCA Feature Space
        df_pca = self.results.get("df_pca")
        if df_pca is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                data=df_pca,
                x="PCA1",
                y="PCA2",
                hue="Class",
                palette="tab10",
                alpha=0.7,
                ax=ax,
            )
            ax.set_title("PCA Feature Space (2D) - Are classes separable?")
            PlotVisualizer.save_and_show(
                fig, "5_pca_feature_space.png", self.save_dir, self.save_reports
            )

        # 6. Sample Grid
        fig = plt.figure(figsize=(15, 10))
        for i, path in enumerate(self.class_paths[:9]):
            files = [
                f
                for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if files:
                img = Image.open(os.path.join(path, files[0]))
                ax = fig.add_subplot(3, 3, i + 1)
                ax.imshow(img)
                ax.set_title(os.path.basename(path))
                ax.axis("off")
        fig.suptitle("Sample Images per Class", fontsize=20)
        PlotVisualizer.save_and_show(
            fig, "6_sample_grid.png", self.save_dir, self.save_reports
        )

        # Save CSV Metrics
        if self.save_reports:
            df.to_csv(
                os.path.join(self.save_dir, "detailed_image_metrics.csv"), index=False
            )
            print(
                f"\n[INFO] All reports and visualizations have been saved in: {self.save_dir}"
            )
