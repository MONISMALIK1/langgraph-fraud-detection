"""Run three sample transactions through the fraud-detection graph.

Works fully offline: without an ANTHROPIC_API_KEY the gray-zone case
falls back to deterministic scoring and lands in the human-review queue.
With a key set, Claude assesses the gray-zone case and its structured
reasoning appears in the report.

    python demo.py
"""
import os

from fraud_agent import build_graph

SCENARIOS = [
    (
        "Regular coffee purchase (expected: APPROVE)",
        {
            "transaction": {
                "transaction_id": "txn_20001",
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
        },
    ),
    (
        "Card-testing burst plus impossible travel (expected: BLOCK)",
        {
            "transaction": {
                "transaction_id": "txn_20002",
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
        },
    ),
    (
        "Large online purchase from a new device (expected: gray zone)",
        {
            "transaction": {
                "transaction_id": "txn_20003",
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
        },
    ),
]


def main() -> None:
    graph = build_graph()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY is not set. Gray-zone cases fall back to")
        print("deterministic scoring and are routed to human review.\n")

    for title, case in SCENARIOS:
        result = graph.invoke(case)
        decision = result["decision"]
        print("=" * 72)
        print(title)
        print("-" * 72)
        txn = case["transaction"]
        print(
            f"  {txn['transaction_id']}  {txn['amount']:.2f} {txn.get('currency', 'USD')}"
            f"  at {txn['merchant']} ({txn['country']})"
        )
        print(f"  Decision:   {decision['action']}   final_score={decision['final_score']}")
        print(f"  Risk band:  {decision['risk_band']}")
        if decision["reasons"]:
            print("  Reasons:")
            for reason in decision["reasons"]:
                print(f"    - {reason}")
        if decision["llm_reasoning"]:
            print(f"  Analyst reasoning: {decision['llm_reasoning']}")
        print("  Audit trail:")
        for line in result["audit_trail"]:
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
