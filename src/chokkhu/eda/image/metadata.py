import os


class MetadataExtractor:
    @staticmethod
    def extract(img_path: str, img_bgr) -> dict:
        """Extracts Topic 1: Metadata and Structural Domain properties."""
        file_size_kb = os.path.getsize(img_path) / 1024.0
        h, w, c = img_bgr.shape
        aspect_ratio = w / h if h > 0 else 0
        total_pixels = w * h

        return {
            "Width": w,
            "Height": h,
            "Aspect_Ratio": aspect_ratio,
            "File_Size_KB": file_size_kb,
            "Total_Pixels": total_pixels,
        }
