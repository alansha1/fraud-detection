"""
Database operations — all queries go here, never in route handlers.
"""

import json
from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas


# ── Transactions ───────────────────────────────────────────────────────────────

def create_transaction(db: Session, txn_in: schemas.TransactionIn,
                       score: dict) -> models.Transaction:
    """Persist the incoming transaction and its fraud score."""
    record = models.Transaction(
        transaction_id    = txn_in.transaction_id,
        amount            = txn_in.amount,
        currency          = txn_in.currency,
        merchant_category = txn_in.merchant_category,
        country           = txn_in.country,
        hour              = txn_in.hour,
        foreign_tx        = txn_in.foreign_tx,
        velocity_1h       = txn_in.velocity_1h,
        amount_vs_avg     = txn_in.amount_vs_avg,
        distance_km       = txn_in.distance_km,
        device_change     = txn_in.device_change,
        fraud_probability = score["fraud_probability"],
        risk_level        = score["risk_level"],
        flagged           = score["flagged"],
        explanation       = json.dumps(score["explanation"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_transaction(db: Session, transaction_id: str) -> Optional[models.Transaction]:
    return db.query(models.Transaction).filter(
        models.Transaction.transaction_id == transaction_id
    ).first()


def list_transactions(db: Session, flagged_only: bool = False,
                      risk_level: Optional[str] = None,
                      skip: int = 0, limit: int = 100) -> List[models.Transaction]:
    q = db.query(models.Transaction)
    if flagged_only:
        q = q.filter(models.Transaction.flagged == True)
    if risk_level:
        q = q.filter(models.Transaction.risk_level == risk_level.upper())
    return q.order_by(models.Transaction.scored_at.desc()).offset(skip).limit(limit).all()


# ── Audit log ──────────────────────────────────────────────────────────────────

def add_audit(db: Session, transaction_id: str, action: str,
              detail: str = "", actor: str = "system") -> models.AuditLog:
    entry = models.AuditLog(
        transaction_id = transaction_id,
        action         = action,
        actor          = actor,
        detail         = detail,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_audit_trail(db: Session, transaction_id: str) -> List[models.AuditLog]:
    return (
        db.query(models.AuditLog)
        .filter(models.AuditLog.transaction_id == transaction_id)
        .order_by(models.AuditLog.created_at.asc())
        .all()
    )


# ── Dashboard stats ────────────────────────────────────────────────────────────

def get_dashboard_stats(db: Session, model_auc: float) -> dict:
    total   = db.query(func.count(models.Transaction.id)).scalar() or 0
    flagged = db.query(func.count(models.Transaction.id)).filter(
        models.Transaction.flagged == True
    ).scalar() or 0
    avg_prob = db.query(func.avg(models.Transaction.fraud_probability)).scalar() or 0.0

    today_start = datetime.combine(date.today(), datetime.min.time())
    high_today  = db.query(func.count(models.Transaction.id)).filter(
        models.Transaction.risk_level == "HIGH",
        models.Transaction.scored_at >= today_start,
    ).scalar() or 0

    return {
        "total_scored":    total,
        "total_flagged":   flagged,
        "fraud_rate_pct":  round((flagged / total * 100) if total else 0, 2),
        "avg_fraud_prob":  round(float(avg_prob), 4),
        "high_risk_today": high_today,
        "model_auc":       model_auc,
    }
