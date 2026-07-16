from datetime import datetime

from fraud_agent.models import CustomerProfile, Transaction
from fraud_agent.rules import evaluate_rules


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


def make_features(**overrides) -> dict:
    base = dict(
        amount_zscore=0.0,
        amount_ratio=1.0,
        velocity_10m=0,
        velocity_1h=0,
        velocity_24h=0,
        small_txn_burst_10m=0,
        travel_speed_kmh=0.0,
        is_new_device=False,
        is_foreign=False,
        is_unusual_category=False,
        is_high_risk_category=False,
        is_night=False,
        is_online=False,
        account_age_days=365,
    )
    base.update(overrides)
    return base


PROFILE = CustomerProfile(customer_id="c1")


def rule_ids(hits):
    return {h["rule_id"] for h in hits}


def test_benign_transaction_hits_nothing():
    hits, score, hard_block = evaluate_rules(make_txn(), PROFILE, make_features())
    assert hits == []
    assert score == 0.0
    assert hard_block is False


def test_blacklist_hard_blocks():
    profile = CustomerProfile(customer_id="c1", is_blacklisted=True)
    hits, _, hard_block = evaluate_rules(make_txn(), profile, make_features())
    assert "R001" in rule_ids(hits)
    assert hard_block is True


def test_impossible_travel_hard_blocks():
    hits, _, hard_block = evaluate_rules(
        make_txn(), PROFILE, make_features(travel_speed_kmh=5000.0)
    )
    assert "R002" in rule_ids(hits)
    assert hard_block is True


def test_card_testing_pattern():
    hits, _, hard_block = evaluate_rules(
        make_txn(amount=1.99), PROFILE, make_features(small_txn_burst_10m=5)
    )
    assert "R003" in rule_ids(hits)
    assert hard_block is True


def test_extreme_amount_is_high_but_not_hard_block():
    hits, score, hard_block = evaluate_rules(
        make_txn(amount=2500.0), PROFILE, make_features(amount_ratio=12.0, amount_zscore=8.0)
    )
    assert "R004" in rule_ids(hits)
    assert hard_block is False
    assert score >= 0.5


def test_new_device_abroad():
    hits, _, hard_block = evaluate_rules(
        make_txn(country="NG", device_id="dev_new"),
        PROFILE,
        make_features(is_new_device=True, is_foreign=True),
    )
    assert "R005" in rule_ids(hits)
    assert hard_block is False


def test_score_is_capped_at_one():
    profile = CustomerProfile(customer_id="c1", is_blacklisted=True)
    _, score, _ = evaluate_rules(
        make_txn(amount=1.0),
        profile,
        make_features(
            travel_speed_kmh=5000.0,
            small_txn_burst_10m=6,
            amount_ratio=20.0,
            velocity_1h=12,
            is_new_device=True,
            is_foreign=True,
        ),
    )
    assert score == 1.0
