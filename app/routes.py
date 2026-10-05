"""
FastAPI route handlers — thin layer that delegates all logic to crud.py and predictor.py.
"""

import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.predictor import get_model_auc, get_model_comparison, score_transaction

router = APIRouter()


# ── Score a transaction ────────────────────────────────────────────────────────

@router.post(
    "/transactions/score",
    response_model=schemas.ScoreResponse,
    status_code=201,
    tags=["Fraud Detection"],
    summary="Score a transaction for fraud risk",
)
def score(txn: schemas.TransactionIn, db: Session = Depends(get_db)):
    """
    Submit a transaction payload and receive an instant fraud risk score.

    Returns fraud probability (0–1), risk level (LOW / MEDIUM / HIGH),
    a flag decision, and an AI explanation of the top contributing factors.
    """
    # Reject duplicate transaction IDs
    if crud.get_transaction(db, txn.transaction_id):
        raise HTTPException(
            status_code=409,
            detail=f"Transaction '{txn.transaction_id}' has already been scored."
        )

    # Run the ML model + SHAP explainer
    result = score_transaction(txn.model_dump())

    # Persist to database
    record = crud.create_transaction(db, txn, result)

    # Write audit entry
    crud.add_audit(
        db,
        txn.transaction_id,
        action="SCORED",
        detail=f"risk={result['risk_level']} prob={result['fraud_probability']}",
    )

    return schemas.ScoreResponse(
        transaction_id    = txn.transaction_id,
        fraud_probability = result["fraud_probability"],
        risk_level        = result["risk_level"],
        flagged           = result["flagged"],
        explanation       = result["explanation"],
        scored_at         = record.scored_at,
    )


# ── Retrieve a scored transaction ──────────────────────────────────────────────

@router.get(
    "/transactions/{transaction_id}",
    response_model=schemas.TransactionOut,
    tags=["Transactions"],
    summary="Get a scored transaction by ID",
)
def get_transaction(transaction_id: str, db: Session = Depends(get_db)):
    """Fetch the stored record for a previously scored transaction."""
    record = crud.get_transaction(db, transaction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return record


# ── Audit trail ────────────────────────────────────────────────────────────────

@router.get(
    "/transactions/{transaction_id}/audit",
    response_model=List[schemas.AuditEntry],
    tags=["Audit"],
    summary="Full audit trail for a transaction",
)
def get_audit(transaction_id: str, db: Session = Depends(get_db)):
    """
    Return the complete compliance audit trail for a transaction —
    every decision, review, or status change in chronological order.
    """
    if not crud.get_transaction(db, transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return crud.get_audit_trail(db, transaction_id)


# ── List transactions ──────────────────────────────────────────────────────────

@router.get(
    "/transactions",
    response_model=List[schemas.TransactionOut],
    tags=["Transactions"],
    summary="List scored transactions",
)
def list_transactions(
    flagged_only: bool  = Query(False, description="Return only flagged transactions"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level: LOW / MEDIUM / HIGH"),
    skip:  int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List all scored transactions with optional filters."""
    return crud.list_transactions(db, flagged_only=flagged_only,
                                  risk_level=risk_level, skip=skip, limit=limit)


# ── Dashboard ──────────────────────────────────────────────────────────────────

@router.get(
    "/dashboard/stats",
    response_model=schemas.DashboardStats,
    tags=["Dashboard"],
    summary="Live fraud detection statistics",
)
def dashboard_stats(db: Session = Depends(get_db)):
    """
    Return real-time statistics: total scored, flagged, fraud rate,
    average fraud probability, high-risk count today, and model AUC.
    """
    return crud.get_dashboard_stats(db, model_auc=get_model_auc())


# ── Model comparison ───────────────────────────────────────────────────────────

@router.get(
    "/models/compare",
    response_model=schemas.ModelComparison,
    tags=["Models"],
    summary="Compare all trained models — pick the best one",
)
def model_comparison():
    """
    Returns AUC-ROC, Precision, Recall, and F1 for every model trained
    during the last training run (Logistic Regression, Decision Tree,
    Random Forest, XGBoost).  Shows which model was auto-selected for
    production and why.

    This endpoint mirrors the model-selection methodology used in the
    training notebook — a standard academic evaluation approach.
    """
    comp = get_model_comparison()
    return schemas.ModelComparison(
        best_model     = comp["best_model"],
        best_auc_roc   = comp["auc_roc"],
        best_f1        = comp["f1"],
        train_samples  = comp["train_samples"],
        fraud_rate_pct = comp["fraud_rate_pct"],
        comparison     = [schemas.ModelMetrics(**m) for m in comp["comparison"]],
    )
