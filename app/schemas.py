"""
Pydantic v2 request / response schemas.
"""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ── Inbound transaction ────────────────────────────────────────────────────────

class TransactionIn(BaseModel):
    """Payload sent to POST /transactions/score."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "transaction_id": "txn_demo_001",
            "amount": 2499.99,
            "currency": "EUR",
            "merchant_category": "electronics",
            "country": "NG",
            "hour": 3,
            "foreign_tx": True,
            "velocity_1h": 7,
            "amount_vs_avg": 4.2,
            "distance_km": 3400,
            "device_change": True,
        }
    })

    transaction_id:     str   = Field(..., description="Unique transaction reference")
    amount:             float = Field(..., gt=0, description="Transaction amount")
    currency:           str   = Field("EUR", description="ISO 4217 currency code")
    merchant_category:  str   = Field("online_retail", description="Merchant category")
    country:            str   = Field("IE", description="ISO 3166-1 alpha-2 country")
    hour:               int   = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    foreign_tx:         bool  = Field(False, description="Is the merchant foreign?")
    velocity_1h:        int   = Field(1, ge=1, description="Transactions in past hour")
    amount_vs_avg:      float = Field(1.0, gt=0, description="Amount ÷ user 30-day avg")
    distance_km:        int   = Field(0, ge=0, description="Distance from home country (km)")
    device_change:      bool  = Field(False, description="New device used?")

    @field_validator("currency")
    @classmethod
    def upper_currency(cls, v: str) -> str:
        return v.upper()

    @field_validator("country")
    @classmethod
    def upper_country(cls, v: str) -> str:
        return v.upper()


# ── Explanation ────────────────────────────────────────────────────────────────

class FraudFactor(BaseModel):
    feature:     str
    description: str
    impact:      str   # "high" | "medium" | "low"


class Explanation(BaseModel):
    top_factors: List[FraudFactor]
    model:       str = "RandomForest v1"


# ── Score response ─────────────────────────────────────────────────────────────

class ScoreResponse(BaseModel):
    transaction_id:    str
    fraud_probability: float
    risk_level:        str
    flagged:           bool
    explanation:       Explanation
    scored_at:         datetime


# ── Transaction record (from DB) ───────────────────────────────────────────────

class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:                 int
    transaction_id:     str
    amount:             float
    currency:           str
    merchant_category:  str
    country:            str
    fraud_probability:  float
    risk_level:         str
    flagged:            bool
    scored_at:          datetime


# ── Audit log ──────────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:             int
    transaction_id: str
    action:         str
    actor:          str
    detail:         Optional[str]
    created_at:     datetime


# ── Dashboard stats ────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_scored:       int
    total_flagged:      int
    fraud_rate_pct:     float
    avg_fraud_prob:     float
    high_risk_today:    int
    model_auc:          float


# ── Model comparison ───────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    """Per-model evaluation metrics from training."""
    model_name: str
    auc_roc:    float
    avg_prec:   float
    precision:  float
    recall:     float
    f1:         float


class ModelComparison(BaseModel):
    """Response for GET /models/compare."""
    best_model:       str
    best_auc_roc:     float
    best_f1:          float
    train_samples:    int
    fraud_rate_pct:   float
    comparison:       List[ModelMetrics]
