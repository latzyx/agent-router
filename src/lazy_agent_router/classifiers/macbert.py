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
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]):
        """一次完成一批文本的分词和 GPU 前向计算。"""
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        # inference_mode 比 no_grad 进一步关闭版本计数；CUDA autocast 减少显存和计算量。
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.device == "cuda",
            ):
                outputs = self.model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )

        scores, indices = torch.max(
            probs,
            dim=-1
        )
        return [
            {
                "intent": self.model.config.id2label[index],
                "confidence": float(score),
            }
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
        ]
