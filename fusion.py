"""Stage 3: combine the two model scores into a final risk assessment."""

from __future__ import annotations

import os

SPOOFING_WEIGHT = float(os.getenv("SPOOFING_WEIGHT", "0.5"))
INTENT_WEIGHT = float(os.getenv("INTENT_WEIGHT", "0.5"))


def fuse_scores(
    spoofing_score: float,
    intent_score: float,
    spoofing_weight: float = SPOOFING_WEIGHT,
    intent_weight: float = INTENT_WEIGHT,
):
    """Fuse normalized spoof and intent scores and classify risk."""
    if not 0 <= spoofing_score <= 1 or not 0 <= intent_score <= 1:
        raise ValueError("Both input scores must be between 0 and 1.")

    total = spoofing_weight + intent_weight
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive.")

    w_spoof = spoofing_weight / total
    w_intent = intent_weight / total
    final_score = (w_spoof * spoofing_score) + (w_intent * intent_score)

    if final_score >= 0.65:
        verdict = "Fraudulent Call Detected (High Risk)"
    elif final_score >= 0.40:
        verdict = "Suspicious Call (Medium Risk)"
    else:
        verdict = "Genuine Call (Low Risk)"

    return final_score, verdict
