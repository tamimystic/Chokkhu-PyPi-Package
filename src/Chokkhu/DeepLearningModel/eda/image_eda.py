import os
import warnings
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)


class ImageEDA:
    def __init__(self, dataset_path: str):
        """
        Initializes the ImageEDA class and triggers the analysis pipeline.
        """
        self.dataset_path: str = dataset_path
        self.results: Dict[str, Any] = {}
        self.class_paths: List[str] = []
        self._perform_eda()

    def _perform_eda(self) -> None:
        """
        Orchestrates the EDA process: Collection, Analysis, and Visualization.
        """
        print(f"--- Executing EDA for: {self.dataset_path} ---")
        self._collect_paths()
        if not self.class_paths:
            print("Error: No valid images found in the specified path.")
            return
        self.results = self._analyze_data()
        self._visual_reports()

    def _collect_paths(self) -> None:
        """
        Identifies class folders containing valid image files.
        """
        for root, _, files in os.walk(self.dataset_path):
            if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files):
                self.class_paths.append(root)

    def _blur_score(self, image: np.ndarray) -> float:
        """
        Estimates image focus using the Variance of Laplacian method.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        score: float = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(score)

    def _analyze_data(self) -> Dict[str, Any]:
        """
        Extracts metrics like image size, aspect ratio, blur score, and RGB histogram.
        """
        exts = (".png", ".jpg", ".jpeg")
        counts = {
            os.path.basename(p): len(
                [f for f in os.listdir(p) if f.lower().endswith(exts)]
            )
            for p in self.class_paths
        }
        df_counts = pd.DataFrame(list(counts.items()), columns=["Class", "Image_Count"])

        sizes: List[Tuple[int, int, float]] = []
        blur_scores: List[float] = []
        total_rgb_hist = np.zeros((256, 3))
        processed_count = 0

        for path in self.class_paths:
            files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
            for img_name in files:
                img_bgr = cv2.imread(os.path.join(path, img_name))
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = img_rgb.shape
                sizes.append((w, h, w / h))
                blur_scores.append(self._blur_score(img_rgb))

                for i in range(3):
                    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                    total_rgb_hist[:, i] += hist.flatten()
                processed_count += 1

        df_sizes = pd.DataFrame(sizes, columns=["Width", "Height", "Aspect_Ratio"])
        avg_hist = (
            total_rgb_hist / processed_count if processed_count > 0 else total_rgb_hist
        )

        return {
            "df_counts": df_counts,
            "sizes_df": df_sizes,
            "total_images": processed_count,
            "avg_blur": np.mean(blur_scores) if blur_scores else 0.0,
            "avg_rgb_hist": avg_hist,
            "total_classes": len(self.class_paths),
        }

    def _visual_reports(self) -> None:
        """
        Displays all visual charts and prints the summary tables.
        """
        res = self.results
        sns.set_theme(style="whitegrid")

        # 1. Distribution Chart
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            data=res["df_counts"], x="Class", y="Image_Count", palette="viridis"
        )
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 8),
                textcoords="offset points",
            )
        plt.title("Class-wise Image Distribution")
        plt.xticks(rotation=45)
        plt.show()

        # 2. Sample Grid
        plt.figure(figsize=(12, 8))
        for i, path in enumerate(self.class_paths[:9]):
            files = [
                f
                for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if files:
                img = Image.open(os.path.join(path, files[0]))
                plt.subplot(3, 3, i + 1)
                plt.imshow(img)
                plt.title(os.path.basename(path))
                plt.axis("off")
        plt.suptitle("Sample Images per Class", fontsize=15)
        plt.tight_layout()
        plt.show()

        # 3. Size Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=res["sizes_df"][["Width", "Height"]], palette="Set2", ax=ax1)
        ax1.set_title("Image Dimensions Outliers")
        sns.histplot(res["sizes_df"]["Aspect_Ratio"], kde=True, color="purple", ax=ax2)
        ax2.set_title("Aspect Ratio Distribution")
        plt.show()

        print("\n" + "-" * 45 + "\nIMAGE SIZE DESCRIPTIVE STATISTICS\n")
        print(res["sizes_df"].describe())
        print("-" * 45)

        # 4. RGB Intensity Distribution
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(["red", "green", "blue"]):
            plt.plot(
                res["avg_rgb_hist"][:, i],
                color=col,
                label=f"{col.upper()} Channel",
                linewidth=1.5,
            )
            plt.fill_between(
                range(256), res["avg_rgb_hist"][:, i], color=col, alpha=0.1
            )
        plt.title("Global Average RGB Intensity Distribution")
        plt.legend()
        plt.show()

        # 5. Final Summary Table
        max_c = res["df_counts"].loc[res["df_counts"]["Image_Count"].idxmax()]
        min_c = res["df_counts"].loc[res["df_counts"]["Image_Count"].idxmin()]
        summary = {
            "EDA Aspect": [
                "Total Classes",
                "Total Images",
                "Majority Class",
                "Minority Class",
                "Avg Aspect Ratio",
                "Avg Blur Score",
            ],
            "Observation": [
                res["total_classes"],
                res["total_images"],
                f"{max_c['Class']} ({max_c['Image_Count']})",
                f"{min_c['Class']} ({min_c['Image_Count']})",
                round(float(res["sizes_df"]["Aspect_Ratio"].mean()), 2),
                round(float(res["avg_blur"]), 2),
            ],
        }
        print("\n" + "-" * 60 + "\nCOMPLETE DATASET EDA SUMMARY\n")
        print(pd.DataFrame(summary).to_string(index=False))
        print("-" * 60 + "\n")
