"""Stage 2: speech transcription and fraudulent-intent recognition."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple

import whisper
from transformers import pipeline

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
# DistilBERT-MNLI is much smaller than BART-large-MNLI while preserving the
# same zero-shot classification approach used by the original project.
INTENT_MODEL = os.getenv(
    "INTENT_MODEL",
    "typeform/distilbert-base-uncased-mnli",
)

LABELS: List[str] = [
    "fraudulent request",
    "scam attempt",
    "asking for sensitive information",
    "urgent money transfer",
    "genuine conversation",
    "normal customer service",
    "casual talk",
]

FRAUD_LABELS = {
    "fraudulent request",
    "scam attempt",
    "asking for sensitive information",
    "urgent money transfer",
}


@lru_cache(maxsize=1)
def _load_whisper(model_size: str):
    return whisper.load_model(model_size)


@lru_cache(maxsize=1)
def _load_intent_classifier(model_id: str):
    return pipeline(
        "zero-shot-classification",
        model=model_id,
        device=-1,
    )


class IntentRecognizer:
    """Reusable Stage-2 model bundle."""

    def __init__(
        self,
        whisper_model_size: str = WHISPER_MODEL_SIZE,
        intent_model: str = INTENT_MODEL,
    ):
        self.whisper_model_size = whisper_model_size
        self.intent_model = intent_model
        self.asr_model = _load_whisper(whisper_model_size)
        self.intent_classifier = _load_intent_classifier(intent_model)

    def transcribe(self, audio_path: str) -> str:
        """Convert speech to text with Whisper."""
        result = self.asr_model.transcribe(audio_path, fp16=False)
        return result.get("text", "").strip()

    def analyze_intent(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Return fraud score plus the label scores produced by the classifier."""
        if not text.strip():
            return 0.0, {}

        result = self.intent_classifier(
            text,
            candidate_labels=LABELS,
            multi_label=False,
        )

        label_scores = {
            label: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }
        fraud_score = min(
            1.0,
            sum(score for label, score in label_scores.items() if label in FRAUD_LABELS),
        )
        return fraud_score, label_scores


def analyze_audio_intent(audio_path: str, recognizer: IntentRecognizer | None = None):
    """Transcribe audio and score its fraudulent intent."""
    recognizer = recognizer or IntentRecognizer()
    transcript = recognizer.transcribe(audio_path)
    score, label_scores = recognizer.analyze_intent(transcript)
    return score, transcript, label_scores


if __name__ == "__main__":
    print("Intent recognition module loaded.")
