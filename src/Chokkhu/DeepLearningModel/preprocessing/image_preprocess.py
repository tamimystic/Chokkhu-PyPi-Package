"""
ImagePreProcessor
A complete dataset preprocessing and EDA pipeline
for pretrained CNN-based transfer learning.

API:
    processor = ImagePreProcessor(datapath="path/to/dataset")
    train, val, test = processor.get_data()
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ImagePreProcessor:
    datapath: str
    img_size: Tuple[int, int]

    raw_images: List[np.ndarray]
    labels: List[int]
    class_names: List[str]
    proc_images: List[np.ndarray]

    X_train: List[np.ndarray]
    X_val: List[np.ndarray]
    X_test: List[np.ndarray]
    y_train: List[int]
    y_val: List[int]
    y_test: List[int]

    def __init__(self, datapath: str) -> None:
        _set_seed(42)

        self.datapath = datapath
        self.img_size = (224, 224)

        self.raw_images, self.labels, self.class_names = self._load_dataset()

        self._eda_before()

        self.proc_images = self._preprocess(self.raw_images)

        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = self._split(self.proc_images, self.labels)

        self._plot_split_distribution(self.y_train, self.y_val, self.y_test)

        y_train_before = self.y_train.copy()

        self.X_train, self.y_train = self._balance_via_augmentation(
            self.X_train, self.y_train
        )

        self._eda_after(
            y_train_before,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def get_data(
        self,
    ) -> Tuple[
        Tuple[List[np.ndarray], List[int]],
        Tuple[List[np.ndarray], List[int]],
        Tuple[List[np.ndarray], List[int]],
    ]:
        return (
            (self.X_train, self.y_train),
            (self.X_val, self.y_val),
            (self.X_test, self.y_test),
        )

    # DATA LOADING

    def _load_dataset(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        images: List[np.ndarray] = []
        labels: List[int] = []

        root = self.datapath
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        if "Training" in subdirs:
            root = os.path.join(root, "Training")

        class_names = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )

        if not class_names:
            raise ValueError("No class folders found in dataset path")

        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                img = cv2.imread(os.path.join(cls_path, f))
                if img is None:
                    continue
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                labels.append(idx)

        if not images:
            raise ValueError("No valid images found in dataset")

        return images, labels, class_names

    # PREPROCESS

    def _preprocess(self, images: List[np.ndarray]) -> List[np.ndarray]:
        processed: List[np.ndarray] = []
        for img in images:
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            processed.append(img)
        return processed

    # SPLIT
    def _split(self, X: List[np.ndarray], y: List[int]) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[int],
        List[int],
        List[int],
    ]:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _plot_split_distribution(
        self, y_train: List[int], y_val: List[int], y_test: List[int]
    ) -> None:
        x = np.arange(len(self.class_names))
        width = 0.25

        plt.figure(figsize=(12, 4))

        for name, labels, offset in [
            ("Train", y_train, -width),
            ("Validation", y_val, 0.0),
            ("Test", y_test, width),
        ]:
            counter = Counter(labels)
            bars = plt.bar(
                x + offset,
                [counter[i] for i in range(len(self.class_names))],
                width,
                label=name,
            )
            for bar in bars:
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    str(int(h)),
                    ha="center",
                    va="bottom",
                )

        plt.xticks(x, self.class_names, rotation=30)
        plt.title("Train / Validation / Test Split Distribution (Stratified)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # AUGMENTATION & BALANCING

    def _augment(self, img: np.ndarray) -> np.ndarray:
        img = img.copy()
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w, _ = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        return img

    def _balance_via_augmentation(
        self, X: List[np.ndarray], y: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        counts: Counter[int] = Counter(y)
        max_count = max(counts.values())

        Xb, yb = list(X), list(y)

        for cls, cnt in counts.items():
            cls_imgs = [X[i] for i in range(len(X)) if y[i] == cls]
            for _ in range(max_count - cnt):
                Xb.append(self._augment(random.choice(cls_imgs)))
                yb.append(cls)

        return Xb, yb

    # EDA HELPERS

    def _geometry_stats(
        self, images: List[np.ndarray]
    ) -> Tuple[List[int], List[int], List[float]]:
        widths, heights, ratios = [], [], []
        for img in images:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
        return widths, heights, ratios

    def _geometry_outliers(self, images: List[np.ndarray], title: str) -> None:
        widths, heights, _ = self._geometry_stats(images)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.boxplot(widths)
        plt.title(f"{title} Width Outliers")
        plt.subplot(1, 2, 2)
        plt.boxplot(heights)
        plt.title(f"{title} Height Outliers")
        plt.tight_layout()
        plt.show()

    def _aspect_ratio_distribution(self, images: List[np.ndarray], title: str) -> None:
        _, _, ratios = self._geometry_stats(images)
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=40)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _rgb_distribution(self, images: List[np.ndarray], title: str) -> None:
        pixels = np.concatenate([img.reshape(-1, 3) for img in images], axis=0)
        plt.figure(figsize=(8, 4))
        plt.hist(pixels[:, 0], bins=50, alpha=0.5, label="R")
        plt.hist(pixels[:, 1], bins=50, alpha=0.5, label="G")
        plt.hist(pixels[:, 2], bins=50, alpha=0.5, label="B")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _blur_score(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    # EDA BEFORE

    def _eda_before(self) -> None:
        counts = Counter(self.labels)
        bars = plt.bar(
            self.class_names,
            [counts[i] for i in range(len(self.class_names))],
        )
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, h, str(h), ha="center", va="bottom"
            )
        plt.title("Class-wise Image Distribution (Before)")
        plt.tight_layout()
        plt.show()

        self._geometry_outliers(self.raw_images, "Before")
        self._aspect_ratio_distribution(
            self.raw_images, "Before Aspect Ratio Distribution"
        )
        self._rgb_distribution(
            self.raw_images, "Before Global RGB Intensity Distribution"
        )

        w, h, r = self._geometry_stats(self.raw_images)
        print("\nIMAGE SIZE DESCRIPTIVE STATISTICS (Before)")
        print(pd.DataFrame({"Width": w, "Height": h, "Aspect_Ratio": r}).describe())

        blur_scores = [self._blur_score(img) for img in self.raw_images]
        self._summary_table(self.raw_images, self.labels, r, blur_scores)

    # EDA AFTER

    def _eda_after(
        self,
        y_train_before: List[int],
        y_train: List[int],
        y_val: List[int],
        y_test: List[int],
    ) -> None:
        before = Counter(y_train_before)
        after = Counter(y_train)

        x = np.arange(len(self.class_names))
        width = 0.35

        plt.figure(figsize=(10, 4))
        bars1 = plt.bar(
            x - width / 2,
            [before[i] for i in range(len(self.class_names))],
            width,
            label="Before Balancing",
        )
        bars2 = plt.bar(
            x + width / 2,
            [after[i] for i in range(len(self.class_names))],
            width,
            label="After Balancing",
        )

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    str(h),
                    ha="center",
                    va="bottom",
                )

        plt.xticks(x, self.class_names, rotation=30)
        plt.legend()
        plt.title("Augmentation-Based Class Balancing (Train)")
        plt.tight_layout()
        plt.show()

        imgs_uint8 = [(img * 255).astype(np.uint8) for img in self.proc_images]

        self._geometry_outliers(imgs_uint8, "After")
        self._aspect_ratio_distribution(imgs_uint8, "After Aspect Ratio Distribution")
        self._rgb_distribution(imgs_uint8, "After Global RGB Intensity Distribution")

        w, h, r = self._geometry_stats(imgs_uint8)
        print("\nIMAGE SIZE DESCRIPTIVE STATISTICS (After)")
        print(pd.DataFrame({"Width": w, "Height": h, "Aspect_Ratio": r}).describe())

        blur_scores = [self._blur_score(img) for img in imgs_uint8]
        self._summary_table(imgs_uint8, self.labels, r, blur_scores)

    # SUMMARY TABLE

    def _summary_table(
        self,
        images: List[np.ndarray],
        labels: List[int],
        ratios: List[float],
        blur_scores: List[float],
    ) -> None:
        counts: Counter[int] = Counter(labels)
        majority = max(counts, key=counts.get)
        minority = min(counts, key=counts.get)

        summary = pd.DataFrame(
            {
                "EDA Aspect": [
                    "Total Classes",
                    "Total Images",
                    "Majority Class",
                    "Minority Class",
                    "Avg Aspect Ratio",
                    "Avg Blur Score",
                ],
                "Observation": [
                    len(self.class_names),
                    len(images),
                    f"{self.class_names[majority]} ({counts[majority]})",
                    f"{self.class_names[minority]} ({counts[minority]})",
                    round(float(np.mean(ratios)), 2),
                    round(float(np.mean(blur_scores)), 2),
                ],
            }
        )

        print("\nDATASET SUMMARY")
        print(summary)
