from __future__ import annotations

import cv2
from skimage.feature import graycomatrix, graycoprops


class TextureAnalyzer:

    @staticmethod
    def extract(gray_img) -> dict:
        edges = cv2.Canny(gray_img, 100, 200).mean()
        glcm = graycomatrix(
            gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
        )
        glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
        glcm_correlation = graycoprops(glcm, "correlation")[0, 0]
        glcm_energy = graycoprops(glcm, "energy")[0, 0]
        glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        return {
            "Edge_Intensity": edges,
            "GLCM_Contrast": glcm_contrast,
            "GLCM_Correlation": glcm_correlation,
            "GLCM_Energy": glcm_energy,
            "GLCM_Homogeneity": glcm_homogeneity,
        }
