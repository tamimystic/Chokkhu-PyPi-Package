import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    def evaluate(self, model, test_ds, class_names):
        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names
            ),
        }
