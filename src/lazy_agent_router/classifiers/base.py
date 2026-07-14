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

    def predict_batch(self, texts: list[str]) -> list[IntentPrediction]:
        """批量分类；未实现向量化的分类器保持兼容并逐条执行。"""
        return [self.predict(text) for text in texts]
