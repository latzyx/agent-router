import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from .base import BaseIntentClassifier
from ..utils.device import resolve_device


class MacBERTClassifier(
    BaseIntentClassifier
):

    def __init__(
            self,
            model_path: str,
            device="auto"
    ):
        self.device = resolve_device(device)

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path,
                local_files_only=True
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_path,
                local_files_only=True
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def predict(
            self,
            text: str
    ):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = (
                self.model(**inputs)
            )

        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )

        score, index = torch.max(
            probs,
            dim=-1
        )

        label = (
            self.model.config
            .id2label[index.item()]
        )

        return {

            "intent": label,

            "confidence":
                float(score.item())
        }
