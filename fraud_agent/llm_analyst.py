"""Claude-powered fraud analyst for gray-zone transactions.

Why an LLM at all: rules and statistics catch known patterns, but the
ambiguous middle band needs contextual reasoning -- weighing a new
device against a plausible merchant, or an unusual amount against a
customer's account age. The analyst receives the full case file
(transaction, features, rule hits, score contributions) and must return
a structured assessment, never free text, so the downstream decision
logic stays deterministic.

If no ANTHROPIC_API_KEY is configured the analyst returns None and the
graph degrades gracefully to its deterministic scores -- availability of
the LLM is never allowed to take the payment pipeline down.
"""
from __future__ import annotations

import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a senior fraud analyst at a payments company.
You review individual card transactions that automated systems flagged as
ambiguous: not clearly legitimate, not clearly fraudulent.

You will receive a case file with the transaction, the customer's
behavioural baseline, engineered risk features, deterministic rule hits,
and statistical score contributions.

Assess the likelihood of fraud. Weigh signals against each other -- a new
device is less alarming for a purchase at the customer's usual merchant;
a modest amount is more alarming during a velocity spike. Be decisive:
recommend APPROVE when the evidence points to the legitimate customer,
BLOCK when it points to fraud, and REVIEW only when a human genuinely
needs to look."""


class LLMAssessment(BaseModel):
    """Structured verdict returned by the analyst."""

    fraud_probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated probability that this transaction is fraudulent."
    )
    key_factors: list[str] = Field(
        description="The 2-5 signals that most influenced the assessment, phrased plainly."
    )
    recommended_action: Literal["APPROVE", "REVIEW", "BLOCK"]
    reasoning: str = Field(description="Short analyst narrative explaining the verdict.")


class ClaudeAnalyst:
    """Callable analyst node backend. Swappable in tests with any
    ``Callable[[dict], Optional[dict]]``."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or os.environ.get("FRAUD_AGENT_MODEL", DEFAULT_MODEL)
        self.temperature = temperature

    def __call__(self, state: dict) -> Optional[dict]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        # Imported lazily so the deterministic pipeline and the test suite
        # never require the API integration to be installed or configured.
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=self.model, temperature=self.temperature, max_tokens=1024)
        structured = llm.with_structured_output(LLMAssessment)
        result = structured.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("user", self._case_file(state)),
            ]
        )
        return result.model_dump()

    @staticmethod
    def _case_file(state: dict) -> str:
        case = {
            "transaction": state["transaction"],
            "customer_profile": {
                k: v
                for k, v in state["customer_profile"].items()
                if k != "recent_transactions"
            },
            "recent_transaction_count": len(
                state["customer_profile"].get("recent_transactions", [])
            ),
            "features": state["features"],
            "rule_hits": state["rule_hits"],
            "rule_score": state["rule_score"],
            "anomaly_score": state["anomaly"],
            "anomaly_contributions": state["anomaly_contributions"],
            "combined_risk": state["combined_risk"],
        }
        return "Case file:\n" + json.dumps(case, indent=2, default=str)
