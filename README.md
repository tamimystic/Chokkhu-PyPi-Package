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
from Chokkhu.DeepLearningModel.eda.image_eda import ImageEDA
from Chokkhu.DeepLearningModel.preprocessing.image_preprocess import ImagePreProcessor

# Dataset EDA
eda = ImageEDA(dataset_path="dataset")

# Dataset preprocessing
processor = ImagePreProcessor(datapath="dataset")
(train_X, train_y), (val_X, val_y), (test_X, test_y) = processor.get_data()


Now you can use it in your model training