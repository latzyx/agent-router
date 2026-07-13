from collections.abc import Sequence

from .base import BaseIntentClassifier, IntentPrediction


class EnsembleIntentClassifier(BaseIntentClassifier):
    """Choose the highest-confidence result from several classifiers."""

    def __init__(self, classifiers: Sequence[BaseIntentClassifier]):
        if not classifiers:
            raise ValueError("at least one classifier is required")
        self.classifiers = list(classifiers)

    def predict(self, text: str) -> IntentPrediction:
        return max((classifier.predict(text) for classifier in self.classifiers), key=lambda item: item["confidence"])
