"""
Test suite for the Fraud Detection API.
Uses in-memory SQLite — no external services needed.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)

# ── Fixtures ───────────────────────────────────────────────────────────────────

HIGH_RISK_TXN = {
    "transaction_id":    "txn_high_001",
    "amount":            4999.99,
    "currency":          "USD",
    "merchant_category": "electronics",
    "country":           "NG",
    "hour":              3,
    "foreign_tx":        True,
    "velocity_1h":       9,
    "amount_vs_avg":     5.5,
    "distance_km":       5000,
    "device_change":     True,
}

LOW_RISK_TXN = {
    "transaction_id":    "txn_low_001",
    "amount":            12.50,
    "currency":          "EUR",
    "merchant_category": "grocery",
    "country":           "IE",
    "hour":              14,
    "foreign_tx":        False,
    "velocity_1h":       1,
    "amount_vs_avg":     0.9,
    "distance_km":       5,
    "device_change":     False,
}

# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "Fraud Detection API"


# ── Scoring ────────────────────────────────────────────────────────────────────

def test_score_high_risk_transaction():
    """High-risk transaction should be flagged with HIGH or MEDIUM risk level."""
    resp = client.post("/api/v1/transactions/score", json=HIGH_RISK_TXN)
    assert resp.status_code == 201
    data = resp.json()
    assert data["transaction_id"] == "txn_high_001"
    assert data["fraud_probability"] >= 0.0
    assert data["risk_level"] in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(data["flagged"], bool)
    assert "explanation" in data
    assert "top_factors" in data["explanation"]
    assert len(data["explanation"]["top_factors"]) > 0


def test_score_low_risk_transaction():
    """Low-risk transaction should have a lower fraud probability."""
    resp = client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    assert resp.status_code == 201
    data = resp.json()
    assert data["transaction_id"] == "txn_low_001"
    # Low-risk should score lower than high-risk
    high_resp = client.post("/api/v1/transactions/score", json={**HIGH_RISK_TXN, "transaction_id": "txn_high_002"})
    assert data["fraud_probability"] < high_resp.json()["fraud_probability"]


def test_duplicate_transaction_rejected():
    """Submitting the same transaction_id twice should return 409."""
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    resp = client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    assert resp.status_code == 409


def test_invalid_amount_rejected():
    """Amount must be positive — zero or negative should return 422."""
    bad = {**LOW_RISK_TXN, "transaction_id": "txn_bad_001", "amount": -50.0}
    resp = client.post("/api/v1/transactions/score", json=bad)
    assert resp.status_code == 422


def test_invalid_hour_rejected():
    """Hour must be 0–23."""
    bad = {**LOW_RISK_TXN, "transaction_id": "txn_bad_002", "hour": 25}
    resp = client.post("/api/v1/transactions/score", json=bad)
    assert resp.status_code == 422


# ── Retrieval ──────────────────────────────────────────────────────────────────

def test_get_transaction():
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    resp = client.get(f"/api/v1/transactions/{LOW_RISK_TXN['transaction_id']}")
    assert resp.status_code == 200
    assert resp.json()["transaction_id"] == LOW_RISK_TXN["transaction_id"]


def test_get_transaction_not_found():
    resp = client.get("/api/v1/transactions/nonexistent_txn")
    assert resp.status_code == 404


# ── Audit trail ────────────────────────────────────────────────────────────────

def test_audit_trail_created_on_score():
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    resp = client.get(f"/api/v1/transactions/{LOW_RISK_TXN['transaction_id']}/audit")
    assert resp.status_code == 200
    logs = resp.json()
    assert len(logs) >= 1
    assert logs[0]["action"] == "SCORED"


def test_audit_trail_not_found():
    resp = client.get("/api/v1/transactions/ghost_txn/audit")
    assert resp.status_code == 404


# ── List & filter ──────────────────────────────────────────────────────────────

def test_list_transactions_empty():
    resp = client.get("/api/v1/transactions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_transactions_after_scoring():
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    client.post("/api/v1/transactions/score", json=HIGH_RISK_TXN)
    resp = client.get("/api/v1/transactions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_flagged_only():
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    client.post("/api/v1/transactions/score", json=HIGH_RISK_TXN)
    resp = client.get("/api/v1/transactions?flagged_only=true")
    assert resp.status_code == 200
    # All returned transactions should be flagged
    for txn in resp.json():
        assert txn["risk_level"] in ("HIGH", "MEDIUM")


# ── Dashboard ──────────────────────────────────────────────────────────────────

def test_dashboard_stats_empty():
    resp = client.get("/api/v1/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_scored"] == 0
    assert data["fraud_rate_pct"] == 0.0
    assert "model_auc" in data


def test_dashboard_stats_after_scoring():
    client.post("/api/v1/transactions/score", json=LOW_RISK_TXN)
    client.post("/api/v1/transactions/score", json=HIGH_RISK_TXN)
    resp = client.get("/api/v1/dashboard/stats")
    data = resp.json()
    assert data["total_scored"] == 2
    assert data["model_auc"] > 0.5


# ── Model comparison ───────────────────────────────────────────────────────────

def test_model_comparison_returns_results():
    resp = client.get("/api/v1/models/compare")
    assert resp.status_code == 200
    data = resp.json()
    assert "best_model" in data
    assert "best_auc_roc" in data
    assert data["best_auc_roc"] > 0.5
    assert isinstance(data["comparison"], list)
    assert len(data["comparison"]) >= 3          # at least LR, DT, RF


def test_model_comparison_has_required_metrics():
    resp = client.get("/api/v1/models/compare")
    data = resp.json()
    for model in data["comparison"]:
        assert "model_name" in model
        assert "auc_roc"    in model
        assert "precision"  in model
        assert "recall"     in model
        assert "f1"         in model
        assert 0.0 <= model["auc_roc"] <= 1.0


def test_model_comparison_best_is_in_list():
    resp = client.get("/api/v1/models/compare")
    data = resp.json()
    names = [m["model_name"] for m in data["comparison"]]
    assert data["best_model"] in names
