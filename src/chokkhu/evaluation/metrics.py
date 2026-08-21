import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1

    return matrix, classes


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> tuple:
    matrix, classes = confusion_matrix(y_true, y_pred)
    n_classes = len(classes)

    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)
    f1s = np.zeros(n_classes)

    for i in range(n_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[i] = (
            2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
            if (precisions[i] + recalls[i]) > 0
            else 0.0
        )

    if average == "macro":
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
    elif average == "weighted":
        weights = np.sum(matrix, axis=1) / np.sum(matrix)
        return (
            float(np.average(precisions, weights=weights)),
            float(np.average(recalls, weights=weights)),
            float(np.average(f1s, weights=weights)),
        )
    else:
        raise ValueError("Unsupported average type. Use 'macro' or 'weighted'.")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))
