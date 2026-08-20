import cv2
import numpy as np


class DuplicateDetector:
    @staticmethod
    def _phash(img_gray):
        resized = cv2.resize(img_gray, (32, 32))
        dct = cv2.dct(np.asarray(resized, dtype=np.float32))
        dctlowfreq = dct[0:8, 0:8]
        med = np.median(dctlowfreq)
        diff = dctlowfreq > med
        return diff.flatten()

    @staticmethod
    def extract(img_gray) -> dict:
        try:
            h = DuplicateDetector._phash(img_gray)
            hash_str = "".join(["1" if b else "0" for b in h])
            return {"pHash": hash_str}
        except Exception:
            return {"pHash": ""}
