"""End-to-end tests for the compiled LangGraph pipeline.

The LLM analyst is injected as a fake, so these tests exercise the full
graph -- ingest, features, rules, scoring, triage, decision -- without
any network access or API key.
"""

from fraud_agent.graph import build_graph


class FakeAnalyst:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return self.assessment


def unavailable_analyst(state):
    return None


def legit_case() -> dict:
    return {
        "transaction": {
            "transaction_id": "txn_legit",
            "customer_id": "cust_001",
            "amount": 8.50,
            "merchant": "Blue Bottle Coffee",
            "merchant_category": "coffee",
            "country": "US",
            "city": "New York",
            "lat": 40.7128,
            "lon": -74.0060,
            "timestamp": "2026-07-16T14:05:00",
            "channel": "card_present",
            "device_id": "dev_sarah_iphone",
        },
        "customer_profile": {
            "customer_id": "cust_001",
            "home_country": "US",
            "avg_transaction_amount": 45.0,
            "std_transaction_amount": 20.0,
            "known_devices": ["dev_sarah_iphone"],
            "usual_categories": ["coffee", "grocery", "restaurants"],
            "recent_transactions": [
                {
                    "timestamp": "2026-07-16T12:00:00",
                    "amount": 32.0,
                    "lat": 40.7128,
                    "lon": -74.0060,
                }
            ],
        },
    }


def fraud_case() -> dict:
    return {
        "transaction": {
            "transaction_id": "txn_fraud",
            "customer_id": "cust_002",
            "amount": 1.99,
            "merchant": "GiftCardHub",
            "merchant_category": "gift_cards",
            "country": "NG",
            "city": "Lagos",
            "lat": 6.5244,
            "lon": 3.3792,
            "timestamp": "2026-07-16T13:59:00",
            "channel": "online",
            "device_id": "dev_unknown_bot",
        },
        "customer_profile": {
            "customer_id": "cust_002",
            "home_country": "US",
            "avg_transaction_amount": 60.0,
            "std_transaction_amount": 30.0,
            "known_devices": ["dev_mark_android"],
            "recent_transactions": [
                {
                    "timestamp": f"2026-07-16T13:5{i}:00",
                    "amount": 1.0 + i * 0.25,
                    "lat": 40.7128,
                    "lon": -74.0060,
                }
                for i in range(2, 7)
            ],
        },
    }


def gray_case() -> dict:
    return {
        "transaction": {
            "transaction_id": "txn_gray",
            "customer_id": "cust_003",
            "amount": 850.0,
            "merchant": "TechDeals Online",
            "merchant_category": "electronics",
            "country": "US",
            "city": "Austin",
            "lat": 30.2672,
            "lon": -97.7431,
            "timestamp": "2026-07-16T15:30:00",
            "channel": "online",
            "device_id": "dev_unknown_win",
        },
        "customer_profile": {
            "customer_id": "cust_003",
            "home_country": "US",
            "avg_transaction_amount": 120.0,
            "std_transaction_amount": 60.0,
            "known_devices": ["dev_omar_mac"],
            "usual_categories": ["electronics", "grocery"],
            "recent_transactions": [
                {
                    "timestamp": "2026-07-15T15:30:00",
                    "amount": 95.0,
                    "lat": 30.2672,
                    "lon": -97.7431,
                }
            ],
        },
    }


def test_legitimate_transaction_is_approved_without_llm():
    analyst = FakeAnalyst({"fraud_probability": 0.9})
    result = build_graph(analyst=analyst).invoke(legit_case())
    assert result["decision"]["action"] == "APPROVE"
    assert result["decision"]["requires_human_review"] is False
    assert analyst.calls == 0


def test_card_testing_with_impossible_travel_is_hard_blocked():
    analyst = FakeAnalyst({"fraud_probability": 0.0})
    result = build_graph(analyst=analyst).invoke(fraud_case())
    decision = result["decision"]
    assert decision["action"] == "BLOCK"
    assert decision["final_score"] == 1.0
    triggered = {h["rule_id"] for h in result["rule_hits"]}
    assert {"R002", "R003"} <= triggered
    # Hard blocks never consult the LLM.
    assert analyst.calls == 0


def test_gray_zone_without_llm_falls_back_to_review():
    result = build_graph(analyst=unavailable_analyst).invoke(gray_case())
    decision = result["decision"]
    assert decision["action"] == "REVIEW"
    assert decision["requires_human_review"] is True
    assert result["llm_assessment"] is None


def test_gray_zone_llm_can_escalate_to_block():
    analyst = FakeAnalyst(
        {
            "fraud_probability": 0.95,
            "key_factors": ["unrecognised device", "amount far above baseline"],
            "recommended_action": "BLOCK",
            "reasoning": "Pattern matches account takeover.",
        }
    )
    result = build_graph(analyst=analyst).invoke(gray_case())
    assert analyst.calls == 1
    assert result["decision"]["action"] == "BLOCK"
    assert result["decision"]["final_score"] >= 0.70


def test_gray_zone_llm_can_clear_to_approve():
    analyst = FakeAnalyst(
        {
            "fraud_probability": 0.05,
            "key_factors": ["merchant matches usual category", "domestic transaction"],
            "recommended_action": "APPROVE",
            "reasoning": "Consistent with an upgrade purchase by the real customer.",
        }
    )
    result = build_graph(analyst=analyst).invoke(gray_case())
    assert analyst.calls == 1
    assert result["decision"]["action"] == "APPROVE"


def test_audit_trail_is_complete_and_ordered():
    result = build_graph(analyst=unavailable_analyst).invoke(gray_case())
    trail = result["audit_trail"]
    assert trail[0].startswith("ingest:")
    assert trail[-1].startswith("decision:")
    stages = [line.split(":")[0] for line in trail]
    assert stages == ["ingest", "features", "rules", "scoring", "llm_analyst", "decision"]
