"""Feature engineering: turn a raw transaction plus customer history
into the numeric and boolean signals the rules and scorer consume.

Everything here is pure and deterministic -- same input, same output --
which is what makes the downstream decisions reproducible and testable.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from .models import CustomerProfile, Transaction

HIGH_RISK_CATEGORIES = {
    "gambling",
    "crypto",
    "gift_cards",
    "wire_transfer",
    "money_transfer",
}

# Fastest commercial flight is ~900 km/h; anything above cannot be the
# same physical card holder.
IMPOSSIBLE_TRAVEL_KMH = 900.0


def _as_naive_datetime(value) -> datetime:
    """Normalise timestamps (str or datetime, aware or naive) to naive UTC-ish
    datetimes so arithmetic never raises on mixed inputs."""
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value))
    return value.replace(tzinfo=None)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometres."""
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def compute_features(txn: Transaction, profile: CustomerProfile) -> dict:
    """Compute all risk signals for one transaction."""
    features: dict = {}
    now = _as_naive_datetime(txn.timestamp)

    # --- Amount signals -------------------------------------------------
    std = max(profile.std_transaction_amount, 1.0)
    features["amount_zscore"] = round((txn.amount - profile.avg_transaction_amount) / std, 3)
    features["amount_ratio"] = round(txn.amount / max(profile.avg_transaction_amount, 1.0), 3)

    # --- Velocity signals (counts of prior transactions in a window) ----
    recent = [
        (_as_naive_datetime(t["timestamp"]), t)
        for t in profile.recent_transactions
    ]
    recent.sort(key=lambda pair: pair[0])

    def count_within(window: timedelta) -> int:
        return sum(1 for ts, _ in recent if timedelta(0) <= now - ts <= window)

    features["velocity_10m"] = count_within(timedelta(minutes=10))
    features["velocity_1h"] = count_within(timedelta(hours=1))
    features["velocity_24h"] = count_within(timedelta(hours=24))
    features["small_txn_burst_10m"] = sum(
        1
        for ts, t in recent
        if timedelta(0) <= now - ts <= timedelta(minutes=10)
        and float(t.get("amount", 0)) < 5.0
    )

    # --- Geo signal: implied travel speed since last located txn --------
    speed_kmh = 0.0
    last_located = next(
        (
            (ts, t)
            for ts, t in reversed(recent)
            if t.get("lat") is not None and t.get("lon") is not None and ts <= now
        ),
        None,
    )
    if last_located and txn.lat is not None and txn.lon is not None:
        last_ts, last_txn = last_located
        distance_km = haversine_km(last_txn["lat"], last_txn["lon"], txn.lat, txn.lon)
        # Floor the elapsed time at one minute so a same-second pair does
        # not divide by zero.
        hours = max((now - last_ts).total_seconds() / 3600.0, 1.0 / 60.0)
        speed_kmh = distance_km / hours
    features["travel_speed_kmh"] = round(speed_kmh, 1)

    # --- Context signals -------------------------------------------------
    features["is_new_device"] = bool(txn.device_id) and txn.device_id not in profile.known_devices
    features["is_foreign"] = txn.country != profile.home_country
    features["is_unusual_category"] = (
        bool(profile.usual_categories) and txn.merchant_category not in profile.usual_categories
    )
    features["is_high_risk_category"] = txn.merchant_category in HIGH_RISK_CATEGORIES
    features["is_night"] = now.hour < 6
    features["is_online"] = txn.channel == "online"
    features["account_age_days"] = profile.account_age_days

    return features
