from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class TextureAnalyzer:

    @staticmethod
    def _glcm_props(gray_img: Any) -> tuple[float, float, float, float]:
        arr = np.asarray(gray_img, dtype=np.uint8)
        if arr.ndim > 2:
            arr = arr[:, :, 0]
        h, w = arr.shape
        if h < 1 or w < 2:
            return 0.0, 1.0, 1.0, 1.0
        i_arr: np.ndarray = arr[:, :-1].ravel().astype(np.int64)
        j_arr: np.ndarray = arr[:, 1:].ravel().astype(np.int64)
        flat_idx_1: np.ndarray = i_arr * 256 + j_arr
        flat_idx_2: np.ndarray = j_arr * 256 + i_arr
        flat_idx: np.ndarray = np.concatenate([flat_idx_1, flat_idx_2])
        counts: np.ndarray = np.bincount(flat_idx, minlength=256 * 256).reshape(
            (256, 256)
        )
        p: np.ndarray = counts.astype(np.float64)
        total = float(p.sum())
        if total > 0:
            p /= total
        i_indices: np.ndarray = np.arange(256, dtype=np.float64).reshape((256, 1))
        j_indices: np.ndarray = np.arange(256, dtype=np.float64).reshape((1, 256))

        contrast = float(np.sum(p * ((i_indices - j_indices) ** 2)))
        homogeneity = float(np.sum(p / (1.0 + np.abs(i_indices - j_indices))))
        energy = float(np.sqrt(np.sum(p**2)))

        mean_i = float(np.sum(i_indices * p))
        mean_j = float(np.sum(j_indices * p))
        var_i = float(np.sum(((i_indices - mean_i) ** 2) * p))
        var_j = float(np.sum(((j_indices - mean_j) ** 2) * p))
        std_i = float(np.sqrt(var_i))
        std_j = float(np.sqrt(var_j))
        if std_i > 0 and std_j > 0:
            correlation = float(
                np.sum(p * (i_indices - mean_i) * (j_indices - mean_j))
                / (std_i * std_j)
            )
        else:
            correlation = 1.0

        return contrast, correlation, energy, homogeneity

    @staticmethod
    def extract(gray_img: Any) -> dict[str, float]:
        edges = float(cv2.Canny(gray_img, 100, 200).mean())
        contrast, correlation, energy, homogeneity = TextureAnalyzer._glcm_props(
            gray_img
        )
        return {
            "Edge_Intensity": edges,
            "GLCM_Contrast": contrast,
            "GLCM_Correlation": correlation,
            "GLCM_Energy": energy,
            "GLCM_Homogeneity": homogeneity,
        }
