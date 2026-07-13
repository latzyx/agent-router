from abc import ABC, abstractmethod
from typing import Any


class BaseEntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> dict[str, Any]:
        """Extract structured parameters from a user utterance."""
        raise NotImplementedError
