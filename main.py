"""Command-line entry point for the dual-stage fraud-call pipeline."""

from __future__ import annotations

import argparse

from anti_spoofing import analyze_audio_spoofing
from fusion import fuse_scores
from intent_recognition import IntentRecognizer, analyze_audio_intent


def process_call(audio_path: str) -> None:
    print(f"\n--- Processing Call: {audio_path} ---")

    print("\n[Stage 1] Detecting AI-generated/spoofed speech...")
    spoofing_score = analyze_audio_spoofing(audio_path)
    print(f"Spoofing Score: {spoofing_score:.4f}")

    print("\n[Stage 2] Transcribing and analyzing intent...")
    recognizer = IntentRecognizer()
    intent_score, transcript, labels = analyze_audio_intent(
        audio_path,
        recognizer=recognizer,
    )
    print(f"Transcript: {transcript!r}")
    print(f"Intent Fraud Score: {intent_score:.4f}")
    print("Intent labels:")
    for label, score in sorted(labels.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {label}: {score:.4f}")

    print("\n[Stage 3] Fusing scores...")
    final_score, verdict = fuse_scores(spoofing_score, intent_score)

    print("-" * 50)
    print(f"Final Risk Score : {final_score:.4f}")
    print(f"Final Verdict    : {verdict}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dual-Stage Fraud Call Detection Pipeline"
    )
    parser.add_argument("audio_path", help="Path to a .wav, .mp3, .flac, .ogg, .m4a or .webm file")
    args = parser.parse_args()
    process_call(args.audio_path)
