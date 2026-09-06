"""Stage 1: AI-generated / spoofed speech detection.

The original repository contained a custom Mel-spectrogram CNN, but it did not
ship trained weights. This module keeps that implementation available as
``anti_spoofing_legacy.py`` and uses a real pretrained Wav2Vec2 anti-spoofing
classifier for the default pipeline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

DEFAULT_MODEL = os.getenv(
    "ANTI_SPOOF_MODEL",
    "Vansh180/deepfake-audio-wav2vec2",
)
TARGET_SR = 16_000
MAX_SECONDS = float(os.getenv("ANTI_SPOOF_MAX_SECONDS", "5"))


def _load_audio(audio_path: str) -> np.ndarray:
    """Load mono audio at 16 kHz and cap it to a fixed duration."""
    audio, _ = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    max_samples = int(TARGET_SR * MAX_SECONDS)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    elif len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))

    return audio.astype(np.float32)


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    """Load the anti-spoofing model once and reuse it across requests."""
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    return extractor, model


def analyze_audio_spoofing(audio_path: str) -> float:
    """Return a spoof probability in [0, 1].

    The selected pretrained model exposes two classes: bonafide and spoof.
    We look up the class index by label rather than assuming an index order.
    """
    audio = _load_audio(audio_path)
    extractor, model = _load_model(DEFAULT_MODEL)

    inputs = extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    labels = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }

    spoof_index = next(
        (
            index
            for index, label in labels.items()
            if "spoof" in label or "fake" in label or "deepfake" in label
        ),
        None,
    )

    if spoof_index is None:
        raise RuntimeError(
            f"Could not find a spoof/fake class in model labels: {labels}"
        )

    return float(probabilities[spoof_index].item())


def get_anti_spoofing_info() -> Dict[str, str]:
    """Return model metadata useful for the API/UI."""
    return {
        "model": DEFAULT_MODEL,
        "sample_rate": str(TARGET_SR),
        "window_seconds": str(MAX_SECONDS),
    }


if __name__ == "__main__":
    print("Anti-spoofing module loaded.")
    print(get_anti_spoofing_info())
