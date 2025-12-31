import tensorflow as tf


class ImagePreprocessor:
    def create_dataset(
        self,
        dataset_path: str,
        image_size,
        batch_size: int,
        shuffle=True,
    ):
        return tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=shuffle,
        )
