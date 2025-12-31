import os


class ImageEDA:
    def analyze(self, dataset_path: str) -> dict:
        class_names = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

        images_per_class = {}
        for cls in class_names:
            cls_path = os.path.join(dataset_path, cls)
            images_per_class[cls] = len(os.listdir(cls_path))

        imbalance = (
            max(images_per_class.values()) / min(images_per_class.values()) > 2
            if len(images_per_class) > 1
            else False
        )

        return {
            "num_classes": len(class_names),
            "class_names": class_names,
            "images_per_class": images_per_class,
            "imbalanced": imbalance,
        }
