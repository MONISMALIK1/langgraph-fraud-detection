"""LangGraph fraud-detection agent: tiered rules + anomaly scoring +
Claude analyst for gray-zone transactions."""

from .graph import FraudState, build_graph
from .llm_analyst import ClaudeAnalyst, LLMAssessment
from .models import CustomerProfile, Transaction

__all__ = [
    "build_graph",
    "FraudState",
    "ClaudeAnalyst",
    "LLMAssessment",
    "Transaction",
    "CustomerProfile",
]

__version__ = "1.0.0"
