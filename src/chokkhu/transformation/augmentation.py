from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ImageAugmenter:

    def __init__(
        self,
        techniques: list[str] | None = None,
        augment_factor: int = 1,
        rotate_range: tuple[int, int] = (-30, 30),
        brightness_range: tuple[float, float] = (0.7, 1.3),
        contrast_range: tuple[float, float] = (0.7, 1.3),
        noise_std: float = 0.05,
        random_state: int | None = None,
    ) -> None:
        self.techniques = techniques or [
            "horizontal_flip",
            "rotate",
            "brightness",
            "contrast",
            "noise",
            "crop",
            "blur",
            "cutout",
        ]
        self.augment_factor = augment_factor
        self.rotate_range = rotate_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.random_state = random_state

    def horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)

    def vertical_flip(self, img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 0)

    def rotate(self, img: np.ndarray, angle: float | None = None) -> np.ndarray:
        h, w = img.shape[:2]
        if angle is None:
            angle = float(np.random.uniform(self.rotate_range[0], self.rotate_range[1]))
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def brightness(self, img: np.ndarray, factor: float | None = None) -> np.ndarray:
        if factor is None:
            factor = float(
                np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            )
        img_f = img.astype(np.float32) * factor
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def contrast(self, img: np.ndarray, factor: float | None = None) -> np.ndarray:
        if factor is None:
            factor = float(
                np.random.uniform(self.contrast_range[0], self.contrast_range[1])
            )
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img_f = mean + factor * (img.astype(np.float32) - mean)
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def noise(self, img: np.ndarray, std: float | None = None) -> np.ndarray:
        if std is None:
            std = self.noise_std
        gauss = np.random.normal(0, std * 255.0, img.shape)
        noisy = img.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def crop(self, img: np.ndarray, crop_pct: float = 0.8) -> np.ndarray:
        h, w = img.shape[:2]
        new_h = int(h * crop_pct)
        new_w = int(w * crop_pct)
        if new_h >= h or new_w >= w or new_h < 1 or new_w < 1:
            return img
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        cropped = img[top : top + new_h, left : left + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def blur(self, img: np.ndarray, ksize: int = 5) -> np.ndarray:
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def cutout(
        self, img: np.ndarray, patch_size: tuple[int, int] = (32, 32)
    ) -> np.ndarray:
        img_res = img.copy()
        h, w = img.shape[:2]
        ph, pw = min(patch_size[0], h // 2), min(patch_size[1], w // 2)
        if ph < 1 or pw < 1:
            return img_res
        y = np.random.randint(0, h - ph)
        x = np.random.randint(0, w - pw)
        img_res[y : y + ph, x : x + pw] = 0
        return img_res

    def mixup(
        self, img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 0.5
        mixed = lam * img1.astype(np.float32) + (1.0 - lam) * img2.astype(np.float32)
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def augment_image(self, img: np.ndarray) -> list[np.ndarray]:
        results = []
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for _ in range(self.augment_factor):
            current = img.copy()
            chosen_tech = np.random.choice(self.techniques)
            if chosen_tech == "horizontal_flip":
                current = self.horizontal_flip(current)
            elif chosen_tech == "vertical_flip":
                current = self.vertical_flip(current)
            elif chosen_tech == "rotate":
                current = self.rotate(current)
            elif chosen_tech == "brightness":
                current = self.brightness(current)
            elif chosen_tech == "contrast":
                current = self.contrast(current)
            elif chosen_tech == "noise":
                current = self.noise(current)
            elif chosen_tech == "crop":
                current = self.crop(current)
            elif chosen_tech == "blur":
                current = self.blur(current)
            elif chosen_tech == "cutout":
                current = self.cutout(current)
            results.append(current)
        return results

    def augment_dataset(
        self, images: list[np.ndarray], labels: list[Any] | None = None
    ) -> tuple[list[np.ndarray], list[Any] | None]:
        aug_images = []
        aug_labels = []

        for i, img in enumerate(images):
            aug_images.append(img)
            lbl = labels[i] if labels is not None else None
            if labels is not None:
                aug_labels.append(lbl)

            generated = self.augment_image(img)
            for gen_img in generated:
                aug_images.append(gen_img)
                if labels is not None:
                    aug_labels.append(lbl)

        return aug_images, (aug_labels if labels is not None else None)
