from collections.abc import Sequence


def accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not labels:
        return 0.0
    return sum(prediction == label for prediction, label in zip(predictions, labels, strict=True)) / len(labels)
