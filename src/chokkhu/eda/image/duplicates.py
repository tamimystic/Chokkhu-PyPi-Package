import cv2
import numpy as np


class DuplicateDetector:
    @staticmethod
    def _phash(img_gray):
        # Resize to 32x32
        resized = cv2.resize(img_gray, (32, 32))
        # DCT
        dct = cv2.dct(np.float32(resized))
        # Get top left 8x8
        dctlowfreq = dct[0:8, 0:8]
        med = np.median(dctlowfreq)
        # Create hash
        diff = dctlowfreq > med
        return diff.flatten()

    @staticmethod
    def extract(img_gray) -> dict:
        """Returns the perceptual hash vector of the image."""
        try:
            h = DuplicateDetector._phash(img_gray)
            # Convert boolean array to hex string for easy comparison
            hash_str = "".join(["1" if b else "0" for b in h])
            return {"pHash": hash_str}
        except Exception:
            return {"pHash": ""}
