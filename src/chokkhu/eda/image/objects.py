import cv2


class ObjectDetector:
    # Use OpenCV's built-in Haar Cascade for frontal face
    # We load it lazily
    _face_cascade = None

    @classmethod
    def extract(cls, img_gray) -> dict:
        """Detects basic objects (like faces) using Haar Cascades to get bounding box counts."""
        try:
            if cls._face_cascade is None:
                # Load default frontal face cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                cls._face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Detect faces
            faces = cls._face_cascade.detectMultiScale(
                img_gray, 
                scaleFactor=1.1, 
                minNeighbors=4, 
                minSize=(30, 30)
            )
            return {"Face_Count": len(faces)}
        except Exception:
            return {"Face_Count": 0}
