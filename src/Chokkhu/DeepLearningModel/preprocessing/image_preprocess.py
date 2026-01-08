"""
ImagePreProcessor
A complete dataset preprocessing and EDA pipeline
for pretrained CNN-based transfer learning.

API:
    train, val, test = ImagePreProcessor(datapath="path/to/dataset")
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split


def _set_seed(seed=42):
    """
    Set global random seed for reproducible EDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ImagePreProcessor:
    """
    ImagePreProcessor loads images from a dataset path, performs
    preprocessing, augmentation-based class balancing, stratified
    train/validation/test split, and comprehensive EDA before and
    after preprocessing.
    """

    def __new__(cls, datapath):
        _set_seed(42)
        self = super().__new__(cls)

        self.datapath = datapath
        self.img_size = (224, 224)

        self.raw_images, self.labels, self.class_names = self._load_dataset()

        self._eda_before()

        self.proc_images = self._preprocess(self.raw_images)

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            self.proc_images,
            self.labels,
            test_size=0.30,
            stratify=self.labels,
            random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=0.50,
            stratify=y_tmp,
            random_state=42
        )

        y_train_before = y_train.copy()

        X_train, y_train = self._balance_via_augmentation(X_train, y_train)

        self._eda_after(y_train_before, y_train, y_val, y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _load_dataset(self):
        """
        Load images from flat or Training/Testing dataset structures.
        """
        images, labels = [], []

        root = self.datapath
        subdirs = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]

        if "Training" in subdirs:
            root = os.path.join(root, "Training")

        class_names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )

        if not class_names:
            raise ValueError("No class folders found in dataset path")

        for idx, cls in enumerate(class_names):
            cls_path = os.path.join(root, cls)
            for f in os.listdir(cls_path):
                path = os.path.join(cls_path, f)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(idx)

        if len(images) == 0:
            raise ValueError("No valid images found in dataset")

        return images, labels, class_names

    def _preprocess(self, images):
        """
        Resize images and normalize pixel values to [0, 1].
        """
        processed = []
        for img in images:
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            processed.append(img)
        return processed

    def _augment(self, img):
        """
        Apply light data augmentation suitable for pretrained CNNs.
        """
        img = img.copy()
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w, _ = img.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
        return img

    def _balance_via_augmentation(self, X, y):
        """
        Balance classes using augmentation on the training set only.
        """
        counts = Counter(y)
        max_count = max(counts.values())

        new_X, new_y = [], []

        for cls, cnt in counts.items():
            cls_imgs = [X[i] for i in range(len(X)) if y[i] == cls]
            need = max_count - cnt
            for _ in range(need):
                new_X.append(self._augment(random.choice(cls_imgs)))
                new_y.append(cls)

        return X + new_X, y + new_y

    def _geometry_stats(self, images):
        """
        Compute width, height, and aspect ratio.
        """
        widths, heights, ratios = [], [], []
        for img in images:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
        return widths, heights, ratios

    def _rgb_distribution(self, images, title):
        """
        Plot global RGB intensity distribution.
        """
        pixels = np.concatenate([img.reshape(-1, 3) for img in images], axis=0)
        plt.figure(figsize=(8, 4))
        plt.hist(pixels[:, 0], bins=50, alpha=0.5, label="R")
        plt.hist(pixels[:, 1], bins=50, alpha=0.5, label="G")
        plt.hist(pixels[:, 2], bins=50, alpha=0.5, label="B")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _geometry_outliers(self, images, title):
        """
        Plot width and height outliers using boxplots.
        """
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

    def _aspect_ratio_distribution(self, images, title):
        """
        Plot aspect ratio distribution.
        """
        _, _, ratios = self._geometry_stats(images)
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=40)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _blur_score(self, img):
        """
        Compute blur score using Laplacian variance.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _eda_before(self):
        """
        Perform EDA on raw images before preprocessing.
        """
        counts = Counter(self.labels)
        values = [counts[i] for i in range(len(self.class_names))]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(self.class_names, values)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h, str(h),
                     ha="center", va="bottom")
        plt.title("Class-wise Image Distribution (Before)")
        plt.tight_layout()
        plt.show()

        self._geometry_outliers(self.raw_images, "Before")
        self._aspect_ratio_distribution(self.raw_images, "Before Aspect Ratio Distribution")
        self._rgb_distribution(self.raw_images, "Before Global RGB Intensity Distribution")

        w, h, r = self._geometry_stats(self.raw_images)
        print("\nIMAGE SIZE DESCRIPTIVE STATISTICS (Before)")
        print(pd.DataFrame({
            "Width": w,
            "Height": h,
            "Aspect_Ratio": r
        }).describe())

        blur_scores = [self._blur_score(img) for img in self.raw_images]
        self._summary_table(self.raw_images, self.labels, r, blur_scores)

    def _eda_after(self, y_train_before, y_train, y_val, y_test):
        """
        Perform EDA after preprocessing, balancing, and splitting.
        """
        before = Counter(y_train_before)
        after = Counter(y_train)

        labels = self.class_names
        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(10, 4))
        b1 = plt.bar(x - width / 2,
                     [before[i] for i in range(len(labels))],
                     width, label="Before Balancing")
        b2 = plt.bar(x + width / 2,
                     [after[i] for i in range(len(labels))],
                     width, label="After Balancing")

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, h, str(h),
                         ha="center", va="bottom")

        plt.xticks(x, labels, rotation=30)
        plt.title("Augmentation-Based Class Balancing (Train)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        train_c = Counter(y_train)
        val_c = Counter(y_val)
        test_c = Counter(y_test)

        plt.figure(figsize=(12, 4))
        for name, counter, offset in [
            ("Train", train_c, -0.25),
            ("Validation", val_c, 0.0),
            ("Test", test_c, 0.25),
        ]:
            bars = plt.bar(
                x + offset,
                [counter[i] for i in range(len(labels))],
                width=0.25,
                label=name
            )
            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, h, str(h),
                         ha="center", va="bottom")

        plt.xticks(x, labels, rotation=30)
        plt.title("Train / Validation / Test Split Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()

        imgs_uint8 = [(img * 255).astype(np.uint8) for img in self.proc_images]

        self._geometry_outliers(imgs_uint8, "After")
        self._aspect_ratio_distribution(imgs_uint8, "After Aspect Ratio Distribution")
        self._rgb_distribution(imgs_uint8, "After Global RGB Intensity Distribution")

        w, h, r = self._geometry_stats(imgs_uint8)
        print("\nIMAGE SIZE DESCRIPTIVE STATISTICS (After)")
        print(pd.DataFrame({
            "Width": w,
            "Height": h,
            "Aspect_Ratio": r
        }).describe())

        blur_scores = [self._blur_score(img) for img in imgs_uint8]
        self._summary_table(imgs_uint8, self.labels, r, blur_scores)

    def _summary_table(self, images, labels, ratios, blur_scores):
        """
        Print a high-level dataset summary table.
        """
        counts = Counter(labels)
        majority = max(counts, key=counts.get)
        minority = min(counts, key=counts.get)

        summary = pd.DataFrame({
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
                round(np.mean(ratios), 2),
                round(np.mean(blur_scores), 2),
            ]
        })

        print("\nDATASET SUMMARY")
        print(summary)
