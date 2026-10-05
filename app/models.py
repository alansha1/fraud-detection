"""
ORM models — one table per domain concept.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, Index
from app.database import Base


class Transaction(Base):
    """Every scored transaction, legitimate or flagged."""
    __tablename__ = "transactions"

    id                  = Column(Integer, primary_key=True, index=True)
    transaction_id      = Column(String(64), unique=True, index=True, nullable=False)
    amount              = Column(Float, nullable=False)
    currency            = Column(String(8), default="EUR")
    merchant_category   = Column(String(64))
    country             = Column(String(8))
    hour                = Column(Integer)
    foreign_tx          = Column(Boolean, default=False)
    velocity_1h         = Column(Integer, default=1)
    amount_vs_avg       = Column(Float, default=1.0)
    distance_km         = Column(Integer, default=0)
    device_change       = Column(Boolean, default=False)

    # Model output
    fraud_probability   = Column(Float, nullable=False)
    risk_level          = Column(String(16), nullable=False)   # LOW / MEDIUM / HIGH
    flagged             = Column(Boolean, nullable=False)
    explanation         = Column(Text)                          # JSON string

    scored_at           = Column(DateTime, default=datetime.utcnow, index=True)


class AuditLog(Base):
    """Immutable record of every API decision for compliance."""
    __tablename__ = "audit_logs"

    id              = Column(Integer, primary_key=True, index=True)
    transaction_id  = Column(String(64), index=True, nullable=False)
    action          = Column(String(32))       # SCORED / REVIEWED / BLOCKED
    actor           = Column(String(64), default="system")
    detail          = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_audit_txn_time", "transaction_id", "created_at"),
    )
