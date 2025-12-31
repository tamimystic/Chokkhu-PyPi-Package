import tensorflow as tf

from Chokkhu.DeepLearningModel.base.base_classifier import BaseImageClassifier
from Chokkhu.DeepLearningModel.eda.image_eda import ImageEDA
from Chokkhu.DeepLearningModel.evaluation.evaluator import Evaluator
from Chokkhu.DeepLearningModel.preprocessing.image_preprocess import ImagePreprocessor
from Chokkhu.DeepLearningModel.training.trainer import Trainer
from Chokkhu.DeepLearningModel.visualization.plots import Plotter


class ConvNextTiny:
    class TransferLearning(BaseImageClassifier):
        def __init__(
            self,
            image_size=(224, 224),
            batch_size=32,
            learning_rate=1e-4,
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=None,
            epochs=10,
            freeze_backbone=True,
        ):
            super().__init__()

            self.image_size = image_size
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.optimizer = optimizer
            self.loss = loss
            self.metrics = metrics or ["accuracy"]
            self.epochs = epochs
            self.freeze_backbone = freeze_backbone

            self.eda = ImageEDA()
            self.preprocessor = ImagePreprocessor()
            self.trainer = Trainer()
            self.evaluator = Evaluator()
            self.plotter = Plotter()

            self.class_names = None

        def _build_model(self, num_classes):
            base_model = tf.keras.applications.ConvNeXtTiny(
                include_top=False,
                weights="imagenet",
                input_shape=(*self.image_size, 3),
                pooling="avg",
            )

            if self.freeze_backbone:
                base_model.trainable = False

            inputs = tf.keras.Input(shape=(*self.image_size, 3))
            x = tf.keras.applications.convnext.preprocess_input(inputs)
            x = base_model(x)
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

            model = tf.keras.Model(inputs, outputs)

            opt = (
                tf.keras.optimizers.Adam(self.learning_rate)
                if self.optimizer == "adam"
                else self.optimizer
            )

            model.compile(
                optimizer=opt,
                loss=self.loss,
                metrics=self.metrics,
            )

            return model

        def Training(self, training_data: str, validation_data: str):
            eda_info = self.eda.analyze(training_data)
            self.class_names = eda_info["class_names"]

            train_ds = self.preprocessor.create_dataset(
                training_data,
                self.image_size,
                self.batch_size,
                shuffle=True,
            )
            val_ds = self.preprocessor.create_dataset(
                validation_data,
                self.image_size,
                self.batch_size,
                shuffle=False,
            )

            self.model = self._build_model(eda_info["num_classes"])
            self.history = self.trainer.fit(self.model, train_ds, val_ds, self.epochs)

        def Testing(self, testing_data: str):
            test_ds = self.preprocessor.create_dataset(
                testing_data,
                self.image_size,
                self.batch_size,
                shuffle=False,
            )

            self.test_results = self.evaluator.evaluate(
                self.model, test_ds, self.class_names
            )

        def output(self):
            self.plotter.plot_history(self.history)
            self.plotter.plot_confusion_matrix(
                self.test_results["confusion_matrix"],
                self.class_names,
            )

            print(self.test_results["classification_report"])
