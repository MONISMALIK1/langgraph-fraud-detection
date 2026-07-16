"""Deterministic rule engine.

Rules encode known fraud patterns as auditable checks. They run before
any statistical model or LLM because a rule hit answers the question a
regulator or chargeback dispute will ask -- "why was this blocked?" --
with a rule ID and a plain-English detail, not a probability.

A ``critical`` hit hard-blocks the transaction with no further analysis.
"""
from __future__ import annotations

from .models import CustomerProfile, Transaction

SEVERITY_WEIGHTS = {"critical": 1.0, "high": 0.5, "medium": 0.25}


def evaluate_rules(
    txn: Transaction, profile: CustomerProfile, features: dict
) -> tuple[list[dict], float, bool]:
    """Return (rule hits, rule score in [0, 1], hard_block flag)."""
    hits: list[dict] = []

    def hit(rule_id: str, name: str, severity: str, detail: str) -> None:
        hits.append(
            {
                "rule_id": rule_id,
                "name": name,
                "severity": severity,
                "weight": SEVERITY_WEIGHTS[severity],
                "detail": detail,
            }
        )

    if profile.is_blacklisted:
        hit("R001", "Blacklisted customer", "critical", "Customer is on the internal blocklist.")

    if features["travel_speed_kmh"] > 900:
        hit(
            "R002",
            "Impossible travel",
            "critical",
            f"Implied travel speed of {features['travel_speed_kmh']:.0f} km/h since the "
            "last located transaction exceeds any commercial flight.",
        )

    if features["small_txn_burst_10m"] >= 4 and txn.amount < 5.0:
        hit(
            "R003",
            "Card testing pattern",
            "critical",
            f"{features['small_txn_burst_10m']} sub-$5 transactions in the last 10 minutes "
            "followed by another micro-charge matches automated card testing.",
        )

    if features["amount_ratio"] >= 10 and txn.amount >= 1000:
        hit(
            "R004",
            "Extreme amount anomaly",
            "high",
            f"Amount is {features['amount_ratio']:.1f}x the customer's average "
            f"({txn.amount:.2f} {txn.currency}).",
        )

    if features["is_new_device"] and features["is_foreign"]:
        hit(
            "R005",
            "New device in foreign country",
            "high",
            f"First sighting of device {txn.device_id} combined with a transaction "
            f"outside the home country ({txn.country}).",
        )

    if features["is_high_risk_category"] and features["amount_zscore"] > 2:
        hit(
            "R006",
            "High-risk merchant with unusual amount",
            "medium",
            f"Category '{txn.merchant_category}' is high-risk and the amount is "
            f"{features['amount_zscore']:.1f} standard deviations above baseline.",
        )

    if features["velocity_1h"] >= 8:
        hit(
            "R007",
            "Transaction velocity spike",
            "high",
            f"{features['velocity_1h']} transactions in the last hour.",
        )

    if features["is_online"] and features["is_new_device"] and features["is_night"]:
        hit(
            "R008",
            "Night-time online purchase from unknown device",
            "medium",
            "Online purchase between 00:00 and 06:00 from a device never seen before.",
        )

    score = min(sum(h["weight"] for h in hits), 1.0)
    hard_block = any(h["severity"] == "critical" for h in hits)
    return hits, round(score, 3), hard_block
