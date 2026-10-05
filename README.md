# Fraud Detection API

A real-time transaction fraud scoring API powered by a **Random Forest** machine learning model with **SHAP-based AI explanations**. Every decision is logged to an immutable audit trail for compliance.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        Client                            │
│          (Bank system / Payment gateway / curl)          │
└──────────────────────────┬───────────────────────────────┘
                           │ POST /transactions/score
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  FastAPI  (main.py)                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │               Router  (routes.py)                  │  │
│  │  POST /transactions/score   ← score a transaction  │  │
│  │  GET  /transactions/{id}    ← retrieve result      │  │
│  │  GET  /transactions/{id}/audit ← compliance log    │  │
│  │  GET  /transactions         ← list / filter        │  │
│  │  GET  /dashboard/stats      ← live KPIs            │  │
│  └───────────────┬────────────────────────────────────┘  │
│                  │                                        │
│  ┌───────────────▼────────────────────────────────────┐  │
│  │           ML Predictor  (predictor.py)             │  │
│  │   RandomForest → fraud probability (0–1)           │  │
│  │   SHAP TreeExplainer → top risk factors            │  │
│  └───────────────┬────────────────────────────────────┘  │
│                  │                                        │
│  ┌───────────────▼────────────────────────────────────┐  │
│  │         CRUD layer  (crud.py)                      │  │
│  │   Persist scores · Audit logs · Dashboard stats    │  │
│  └───────────────┬────────────────────────────────────┘  │
│                  │  SQLAlchemy ORM                        │
│  ┌───────────────▼────────────────────────────────────┐  │
│  │       Database  (SQLite dev / PostgreSQL prod)      │  │
│  │   transactions ── audit_logs                        │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI 0.115+ |
| ML model | scikit-learn RandomForest |
| AI explainability | SHAP (TreeExplainer) |
| ORM | SQLAlchemy 2.0 |
| Validation | Pydantic v2 |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Tests | pytest + FastAPI TestClient |
| Deploy | Google Cloud Run / Render |

---

## Quick Start

```bash
git clone https://github.com/alansha1/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt

# Train the model (run once)
python model/train.py

# Start the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### Score a transaction

```bash
POST /api/v1/transactions/score
```

```json
{
  "transaction_id": "txn_001",
  "amount": 2499.99,
  "currency": "EUR",
  "merchant_category": "electronics",
  "country": "NG",
  "hour": 3,
  "foreign_tx": true,
  "velocity_1h": 7,
  "amount_vs_avg": 4.2,
  "distance_km": 3400,
  "device_change": true
}
```

**Response:**

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.94,
  "risk_level": "HIGH",
  "flagged": true,
  "explanation": {
    "top_factors": [
      {
        "feature": "distance_km",
        "description": "Distance from home country: 3400 (increases fraud risk)",
        "impact": "high"
      },
      {
        "feature": "amount_vs_avg",
        "description": "Amount vs. user 30-day average: 4.2 (increases fraud risk)",
        "impact": "high"
      },
      {
        "feature": "velocity_1h",
        "description": "Transactions in past hour: 7 (increases fraud risk)",
        "impact": "medium"
      },
      {
        "feature": "hour",
        "description": "Hour of day: 3 (increases fraud risk)",
        "impact": "medium"
      }
    ],
    "model": "RandomForest v1"
  },
  "scored_at": "2026-10-05T12:00:00"
}
```

### Other endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/transactions/{id}` | Get a scored transaction |
| `GET` | `/api/v1/transactions/{id}/audit` | Full compliance audit trail |
| `GET` | `/api/v1/transactions` | List all transactions (filter: `?flagged_only=true&risk_level=HIGH`) |
| `GET` | `/api/v1/dashboard/stats` | Live KPIs: fraud rate, model AUC, high-risk today |

---

## AI Explanation (SHAP)

This API uses **SHAP (SHapley Additive exPlanations)** — the industry standard for ML model explainability. Each score response includes the top 4 features that most influenced the fraud decision, along with whether they increased or decreased the risk. This is critical for:

- **Compliance** — regulators require explainable AI decisions
- **Analyst review** — fraud teams need to know *why* a transaction was flagged
- **Customer disputes** — explainable decisions support fair outcomes

---

## Tests

```bash
pytest tests/ -v
# 16 passed
```

---

## Deployment (Google Cloud Run)

```bash
gcloud run deploy fraud-detection \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## Project Structure

```
fraud-detection/
├── app/
│   ├── main.py        # FastAPI app entry point
│   ├── database.py    # SQLAlchemy engine + session
│   ├── models.py      # ORM models
│   ├── schemas.py     # Pydantic v2 schemas
│   ├── crud.py        # Database operations
│   ├── routes.py      # API endpoints
│   └── predictor.py   # ML scoring + SHAP explainer
├── model/
│   ├── train.py       # Model training script
│   └── model_meta.json
├── tests/
│   └── test_api.py    # 16 test cases
├── requirements.txt
├── .env.example
└── README.md
```
