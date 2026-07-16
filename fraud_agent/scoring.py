"""Statistical anomaly scoring.

A fast, dependency-free ensemble over the engineered features. Each
component contributes a value in [0, 1]; the weighted sum is the anomaly
score. Rule score and anomaly score are then fused with a noisy-OR,
which is the standard way to combine independent evidence of the same
event: risk only compounds, and either signal alone can raise the total.
"""
from __future__ import annotations

import math

WEIGHTS = {
    "amount": 0.30,
    "velocity": 0.20,
    "geo": 0.20,
    "device": 0.15,
    "context": 0.15,
}

# Combined-risk thresholds shared by triage and the final decision.
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.80


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def anomaly_score(features: dict) -> tuple[float, dict]:
    """Return (score in [0, 1], per-component contributions)."""
    contributions: dict = {}

    # Amounts only start contributing past ~3 standard deviations, so
    # normal week-to-week spending variation scores near zero.
    contributions["amount"] = round(_sigmoid(abs(features["amount_zscore"]) - 3.0), 3)

    contributions["velocity"] = round(min(features["velocity_1h"] / 10.0, 1.0), 3)

    contributions["geo"] = round(min(features["travel_speed_kmh"] / 1000.0, 1.0), 3)

    contributions["device"] = 1.0 if features["is_new_device"] else 0.0

    context = (
        0.35 * features["is_foreign"]
        + 0.25 * features["is_unusual_category"]
        + 0.20 * features["is_high_risk_category"]
        + 0.20 * features["is_night"]
    )
    contributions["context"] = round(min(context, 1.0), 3)

    score = sum(WEIGHTS[name] * value for name, value in contributions.items())
    return round(min(score, 1.0), 3), contributions


def combined_risk(anomaly: float, rule_score: float) -> float:
    """Noisy-OR fusion: P(risk) = 1 - (1 - anomaly)(1 - rules)."""
    return round(1.0 - (1.0 - anomaly) * (1.0 - rule_score), 3)


def risk_band(combined: float) -> str:
    if combined < LOW_RISK_THRESHOLD:
        return "low"
    if combined < HIGH_RISK_THRESHOLD:
        return "elevated"
    return "high"
