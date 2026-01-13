# Chokkhu

Chokkhu is a deep learning image dataset EDA and preprocessing toolkit designed to
prepare train-ready image data for TensorFlow / Keras–based image classification.
It helps users analyze datasets, preprocess images, handle class imbalance, and
train deep learning models using a clean and reproducible pipeline. The package
follows industry-grade Python packaging standards, supports CI/CD pipelines, and
works seamlessly in Google Colab and Jupyter Notebook environments.

INSTALLATION (IMPORTANT – FIRST STEP)

>>> pip install chokkhu <<<

TensorFlow is installed automatically as a runtime dependency.

Chokkhu’s main responsibility is data preparation. It performs image exploratory
data analysis (EDA), class-wise distribution visualization, image size, aspect
ratio, RGB intensity and blur analysis, standard preprocessing (resize to 224×224
and normalization), stratified train/validation/test splitting, and automatic
class balancing using data augmentation. After this step, the dataset is fully
ready to be used for training any deep learning model.

Complete usage example showing the full workflow in one place:

```python

from Chokkhu.DeepLearningModel.EDA import ImageEDA
from Chokkhu.DeepLearningModel.PreProcessing import ImagePreProcessor

# Dataset EDA
eda = ImageEDA(dataset_path="your_dataset_path")

# Dataset preprocessing
processor = ImagePreProcessor(datapath="your_dataset_path")
(train_X, train_y), (val_X, val_y), (test_X, test_y) = processor.get_data()




After excecuting this, You can train your model like this.







import tensorflow as tf
# Example 1: Custom CNN (from scratch)
custom_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

custom_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

custom_model.fit(
    train_X,
    train_y,
    validation_data=(val_X, val_y),
    epochs=10
)




# Example 2: Transfer Learning with ConvNeXt-Tiny (frozen backbone)
import tensorflow as tf
base_model = tf.keras.applications.ConvNeXtTiny(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

transfer_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

transfer_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

transfer_model.fit(
    train_X,
    train_y,
    validation_data=(val_X, val_y),
    epochs=5
)




# Example 3: Fine-tuning ConvNeXt-Tiny (unfrozen backbone)
import tensorflow as tf
base_model.trainable = True

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

transfer_model.fit(
    train_X,
    train_y,
    validation_data=(val_X, val_y),
    epochs=5
)
