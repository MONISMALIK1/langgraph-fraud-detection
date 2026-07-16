"""Data models for transactions and customer profiles.

Pydantic models validate typed inputs at the graph boundary, so a
malformed transaction fails fast at ingest instead of producing a
confusing error deep inside a node.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """A single payment event to be scored."""

    transaction_id: str
    customer_id: str
    amount: float = Field(gt=0)
    currency: str = "USD"
    merchant: str
    merchant_category: str = "general"
    country: str
    city: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp: datetime
    channel: str = "card_present"  # card_present | online
    device_id: Optional[str] = None
    ip_address: Optional[str] = None


class CustomerProfile(BaseModel):
    """Behavioural baseline for the customer, supplied by the caller.

    In production this would come from a feature store; here it travels
    with the request so the graph stays self-contained and testable.
    Each entry in ``recent_transactions`` is a dict with at least a
    ``timestamp`` (ISO string or datetime) and ``amount``, optionally
    ``lat``/``lon``/``country``.
    """

    customer_id: str
    home_country: str = "US"
    avg_transaction_amount: float = 50.0
    std_transaction_amount: float = 25.0
    known_devices: list[str] = Field(default_factory=list)
    usual_categories: list[str] = Field(default_factory=list)
    recent_transactions: list[dict] = Field(default_factory=list)
    account_age_days: int = 365
    is_blacklisted: bool = False
