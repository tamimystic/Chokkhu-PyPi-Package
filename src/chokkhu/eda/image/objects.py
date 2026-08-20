from __future__ import annotations

import cv2


class ObjectDetector:
    _face_cascade = None

    @classmethod
    def extract(cls, img_gray) -> dict:
        try:
            if cls._face_cascade is None:
                haarcascades = getattr(getattr(cv2, "data", None), "haarcascades", "")
                cascade_path = haarcascades + "haarcascade_frontalface_default.xml"
                cls._face_cascade = getattr(cv2, "CascadeClassifier")(cascade_path)
            faces = cls._face_cascade.detectMultiScale(
                img_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            return {"Face_Count": len(faces)}
        except Exception:
            return {"Face_Count": 0}
