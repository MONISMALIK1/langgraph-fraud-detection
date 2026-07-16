from datetime import datetime

from fraud_agent.features import compute_features, haversine_km
from fraud_agent.models import CustomerProfile, Transaction


def make_txn(**overrides) -> Transaction:
    base = dict(
        transaction_id="t1",
        customer_id="c1",
        amount=50.0,
        merchant="Shop",
        merchant_category="grocery",
        country="US",
        timestamp=datetime(2026, 7, 16, 14, 0, 0),
        device_id="dev_known",
    )
    base.update(overrides)
    return Transaction(**base)


def make_profile(**overrides) -> CustomerProfile:
    base = dict(
        customer_id="c1",
        home_country="US",
        avg_transaction_amount=50.0,
        std_transaction_amount=20.0,
        known_devices=["dev_known"],
    )
    base.update(overrides)
    return CustomerProfile(**base)


def test_haversine_nyc_to_london():
    distance = haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
    assert abs(distance - 5570) < 60


def test_amount_zscore():
    features = compute_features(make_txn(amount=90.0), make_profile())
    assert features["amount_zscore"] == 2.0


def test_impossible_travel_speed():
    profile = make_profile(
        recent_transactions=[
            {
                "timestamp": "2026-07-16T13:30:00",
                "amount": 20.0,
                "lat": 40.7128,
                "lon": -74.0060,
            }
        ]
    )
    txn = make_txn(lat=51.5074, lon=-0.1278, timestamp=datetime(2026, 7, 16, 14, 0, 0))
    features = compute_features(txn, profile)
    # ~5570 km in 30 minutes is ~11,000 km/h.
    assert features["travel_speed_kmh"] > 900


def test_velocity_windows_and_small_burst():
    recent = [
        {"timestamp": f"2026-07-16T13:5{i}:00", "amount": 1.5} for i in range(5)
    ] + [{"timestamp": "2026-07-16T09:00:00", "amount": 80.0}]
    features = compute_features(
        make_txn(timestamp=datetime(2026, 7, 16, 14, 0, 0)),
        make_profile(recent_transactions=recent),
    )
    assert features["velocity_10m"] == 5
    assert features["velocity_1h"] == 5
    assert features["velocity_24h"] == 6
    assert features["small_txn_burst_10m"] == 5


def test_device_and_context_flags():
    features = compute_features(
        make_txn(device_id="dev_never_seen", country="NG", merchant_category="gambling"),
        make_profile(usual_categories=["grocery"]),
    )
    assert features["is_new_device"] is True
    assert features["is_foreign"] is True
    assert features["is_unusual_category"] is True
    assert features["is_high_risk_category"] is True
