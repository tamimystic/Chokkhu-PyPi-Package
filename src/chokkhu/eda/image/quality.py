import cv2
from skimage.measure import shannon_entropy


class QualityAnalyzer:
    @staticmethod
    def extract(gray_img, brightness: float, contrast: float) -> dict:
        """Extracts Topic 4: Quality and Information Theory properties."""
        blur = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        entropy = shannon_entropy(gray_img)
        snr = brightness / contrast if contrast > 0 else 0

        return {"Blur_Score": blur, "Shannon_Entropy": entropy, "SNR": snr}
