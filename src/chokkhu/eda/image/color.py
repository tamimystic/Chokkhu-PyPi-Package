import numpy as np
from scipy.stats import skew


class ColorAnalyzer:
    @staticmethod
    def extract(gray_img) -> dict:
        """Extracts Topic 2: Statistical and Color Domain properties."""
        brightness = gray_img.mean()
        contrast = gray_img.std()

        # We catch warnings or handle zeros if perfectly flat image
        if gray_img.size == 0:
            gray_skew = 0.0
        else:
            gray_skew = skew(gray_img.flatten())

        overexposed = np.sum(gray_img >= 250) / gray_img.size * 100
        underexposed = np.sum(gray_img <= 5) / gray_img.size * 100

        return {
            "Brightness": brightness,
            "Contrast": contrast,
            "Skewness": float(gray_skew),
            "Overexposed_%": overexposed,
            "Underexposed_%": underexposed,
        }
