from fraud_agent.scoring import anomaly_score, combined_risk, risk_band


def make_features(**overrides) -> dict:
    base = dict(
        amount_zscore=0.0,
        velocity_1h=0,
        travel_speed_kmh=0.0,
        is_new_device=False,
        is_foreign=False,
        is_unusual_category=False,
        is_high_risk_category=False,
        is_night=False,
    )
    base.update(overrides)
    return base


def test_benign_features_score_low():
    score, contributions = anomaly_score(make_features())
    assert score < 0.1
    assert all(0.0 <= v <= 1.0 for v in contributions.values())


def test_anomalous_features_score_high():
    score, _ = anomaly_score(
        make_features(
            amount_zscore=10.0,
            velocity_1h=10,
            travel_speed_kmh=2000.0,
            is_new_device=True,
            is_foreign=True,
            is_night=True,
        )
    )
    assert score > 0.8


def test_noisy_or_properties():
    assert combined_risk(0.0, 0.0) == 0.0
    assert combined_risk(1.0, 0.3) == 1.0
    assert combined_risk(0.3, 1.0) == 1.0
    # Either signal alone raises the total; together they compound.
    assert combined_risk(0.5, 0.5) == 0.75
    assert combined_risk(0.5, 0.0) == 0.5


def test_risk_bands():
    assert risk_band(0.29) == "low"
    assert risk_band(0.30) == "elevated"
    assert risk_band(0.79) == "elevated"
    assert risk_band(0.80) == "high"
