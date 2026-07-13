"""Small deterministic baseline classifier used when a fine-tuned model is absent."""

from collections.abc import Mapping, Sequence

from .base import BaseIntentClassifier, IntentPrediction


class KeywordIntentClassifier(BaseIntentClassifier):
    """Match configured Chinese keywords and return a calibrated fallback result.

    This is deliberately simple: it makes the service usable in a fresh WSL
    checkout while MacBERT remains the production classifier.
    """

    def __init__(self, intent_keywords: Mapping[str, Sequence[str]], fallback_intent: str = "unknown"):
        self.intent_keywords = {
            intent: tuple(keyword.lower() for keyword in keywords)
            for intent, keywords in intent_keywords.items()
        }
        self.fallback_intent = fallback_intent

    def predict(self, text: str) -> IntentPrediction:
        normalized = text.strip().lower()
        if not normalized:
            return {"intent": self.fallback_intent, "confidence": 0.0}

        scored = []
        for intent, keywords in self.intent_keywords.items():
            matches = sum(keyword in normalized for keyword in keywords if keyword)
            if matches:
                # Several independent keyword hits are more trustworthy, but
                # retain some uncertainty so policy thresholds remain useful.
                score = min(0.55 + 0.15 * matches, 0.95)
                scored.append((score, intent))
        if not scored:
            return {"intent": self.fallback_intent, "confidence": 0.0}
        score, intent = max(scored, key=lambda value: (value[0], value[1]))
        return {"intent": intent, "confidence": score}
