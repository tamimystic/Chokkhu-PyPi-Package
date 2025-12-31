import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def plot_history(self, history):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss")

        plt.show()

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
