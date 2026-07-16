"""LangGraph pipeline assembly.

Why a graph instead of one prompt: fraud decisions must be fast, cheap,
and auditable for the clear majority of cases, and smart for the
ambiguous minority. The graph runs the deterministic tiers first and
routes only the gray zone to the LLM analyst.

    ingest -> features -> rules -> scoring --(triage)--> decide
                                       \\--(gray zone)--> llm_analyst -> decide

State flows through typed nodes; ``audit_trail`` uses an additive
reducer so every node appends its own line and the final state carries a
complete, ordered account of how the decision was reached.
"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .features import compute_features
from .llm_analyst import ClaudeAnalyst
from .models import CustomerProfile, Transaction
from .rules import evaluate_rules
from .scoring import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    anomaly_score,
    combined_risk,
    risk_band,
)

# Weighting between the deterministic combined risk and the LLM's
# calibrated probability when both are available.
DETERMINISTIC_WEIGHT = 0.45
LLM_WEIGHT = 0.55

APPROVE_BELOW = 0.30
BLOCK_AT_OR_ABOVE = 0.70


class FraudState(TypedDict, total=False):
    transaction: dict
    customer_profile: dict
    features: dict
    rule_hits: list
    rule_score: float
    hard_block: bool
    anomaly: float
    anomaly_contributions: dict
    combined_risk: float
    risk_band: str
    llm_assessment: Optional[dict]
    decision: dict
    audit_trail: Annotated[list, operator.add]


AnalystFn = Callable[[dict], Optional[dict]]


def build_graph(analyst: Optional[AnalystFn] = None, checkpointer=None):
    """Compile the fraud-detection graph.

    ``analyst`` is any callable taking the graph state and returning an
    assessment dict (or None). It defaults to :class:`ClaudeAnalyst`;
    tests inject fakes, and other deployments can plug in a different
    model without touching the graph.
    """
    analyst = analyst if analyst is not None else ClaudeAnalyst()

    def ingest(state: FraudState) -> dict:
        txn = Transaction.model_validate(state["transaction"])
        profile = CustomerProfile.model_validate(state["customer_profile"])
        return {
            "transaction": txn.model_dump(),
            "customer_profile": profile.model_dump(),
            "audit_trail": [f"ingest: validated transaction {txn.transaction_id}"],
        }

    def engineer_features(state: FraudState) -> dict:
        txn = Transaction.model_validate(state["transaction"])
        profile = CustomerProfile.model_validate(state["customer_profile"])
        features = compute_features(txn, profile)
        return {
            "features": features,
            "audit_trail": [f"features: computed {len(features)} signals"],
        }

    def run_rules(state: FraudState) -> dict:
        txn = Transaction.model_validate(state["transaction"])
        profile = CustomerProfile.model_validate(state["customer_profile"])
        hits, score, hard_block = evaluate_rules(txn, profile, state["features"])
        return {
            "rule_hits": hits,
            "rule_score": score,
            "hard_block": hard_block,
            "audit_trail": [
                f"rules: {len(hits)} hit(s) [{', '.join(h['rule_id'] for h in hits) or 'none'}], "
                f"score={score:.2f}, hard_block={hard_block}"
            ],
        }

    def score(state: FraudState) -> dict:
        anomaly, contributions = anomaly_score(state["features"])
        combined = combined_risk(anomaly, state["rule_score"])
        band = risk_band(combined)
        return {
            "anomaly": anomaly,
            "anomaly_contributions": contributions,
            "combined_risk": combined,
            "risk_band": band,
            "audit_trail": [
                f"scoring: anomaly={anomaly:.2f}, combined={combined:.2f}, band={band}"
            ],
        }

    def triage(state: FraudState) -> str:
        """Route: hard blocks and clear cases skip the LLM entirely."""
        if state["hard_block"]:
            return "decide"
        if LOW_RISK_THRESHOLD <= state["combined_risk"] < HIGH_RISK_THRESHOLD:
            return "llm_analyst"
        return "decide"

    def llm_analyst_node(state: FraudState) -> dict:
        assessment = analyst(dict(state))
        note = (
            "llm_analyst: structured assessment received"
            if assessment
            else "llm_analyst: unavailable, falling back to deterministic scores"
        )
        return {"llm_assessment": assessment, "audit_trail": [note]}

    def decide(state: FraudState) -> dict:
        combined = state["combined_risk"]
        llm = state.get("llm_assessment")
        reasons = [f"{h['rule_id']} {h['name']}: {h['detail']}" for h in state["rule_hits"]]

        if state["hard_block"]:
            action, final = "BLOCK", 1.0
            reasons.insert(0, "Critical rule triggered: automatic block, no model consulted.")
        else:
            final = combined
            if llm:
                final = round(
                    DETERMINISTIC_WEIGHT * combined + LLM_WEIGHT * llm["fraud_probability"], 3
                )
                reasons.extend(llm["key_factors"])
            if final < APPROVE_BELOW:
                action = "APPROVE"
            elif final < BLOCK_AT_OR_ABOVE:
                action = "REVIEW"
            else:
                action = "BLOCK"
            # The LLM can escalate but never single-handedly approve past
            # the deterministic evidence.
            if llm and llm["recommended_action"] == "BLOCK" and action == "APPROVE":
                action = "REVIEW"
                reasons.append("LLM analyst recommended BLOCK; escalated to human review.")

        decision = {
            "transaction_id": state["transaction"]["transaction_id"],
            "action": action,
            "final_score": round(final, 3),
            "risk_band": state["risk_band"],
            "reasons": reasons,
            "llm_reasoning": llm["reasoning"] if llm else None,
            "requires_human_review": action == "REVIEW",
        }
        return {
            "decision": decision,
            "audit_trail": [f"decision: {action} (final_score={final:.2f})"],
        }

    graph = StateGraph(FraudState)
    graph.add_node("ingest", ingest)
    graph.add_node("features", engineer_features)
    graph.add_node("rules", run_rules)
    graph.add_node("scoring", score)
    graph.add_node("llm_analyst", llm_analyst_node)
    graph.add_node("decide", decide)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "features")
    graph.add_edge("features", "rules")
    graph.add_edge("rules", "scoring")
    graph.add_conditional_edges(
        "scoring", triage, {"llm_analyst": "llm_analyst", "decide": "decide"}
    )
    graph.add_edge("llm_analyst", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=checkpointer)
