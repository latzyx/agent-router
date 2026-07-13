from abc import ABC, abstractmethod
from typing import TypedDict


class IntentPrediction(TypedDict):
    intent: str
    confidence: float


class BaseIntentClassifier(ABC):


    @abstractmethod
    def predict(
        self,
        text: str
    ) -> IntentPrediction:
        """Classify one non-empty user utterance."""
        raise NotImplementedError
