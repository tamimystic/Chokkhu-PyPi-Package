from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class QualityAnalyzer:

    @staticmethod
    def _entropy(image: Any) -> float:
        flat = np.asarray(image, dtype=np.int64).ravel()
        counts = np.bincount(flat)
        probs = counts[counts > 0] / float(flat.size)
        log_probs = np.log2(probs)
        return float(-1.0 * np.sum(probs * log_probs))

    @staticmethod
    def extract(gray_img: Any, brightness: float, contrast: float) -> dict[str, float]:
        blur = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
        entropy = QualityAnalyzer._entropy(gray_img)
        snr = float(brightness / contrast) if contrast > 0 else 0.0
        return {"Blur_Score": blur, "Shannon_Entropy": entropy, "SNR": snr}
