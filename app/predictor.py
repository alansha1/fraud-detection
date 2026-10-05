"""
ML predictor — loads the trained model once at startup and serves predictions.
Uses SHAP TreeExplainer to produce human-readable fraud explanations.
"""

import json
import os
import pickle
from functools import lru_cache
from typing import Any

import numpy as np
import shap

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "fraud_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "model_meta.json")

# Human-readable descriptions for each feature
FEATURE_DESCRIPTIONS = {
    "amount":           "Transaction amount",
    "hour":             "Hour of day",
    "foreign_tx":       "Foreign merchant",
    "merchant_category":"Merchant category",
    "currency":         "Transaction currency",
    "country":          "Transaction country",
    "velocity_1h":      "Transactions in past hour",
    "amount_vs_avg":    "Amount vs. user 30-day average",
    "distance_km":      "Distance from home country",
    "device_change":    "New device used",
}


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """Load model from disk once; cache in memory for all subsequent requests."""
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    with open(META_PATH) as f:
        payload["meta"] = json.load(f)
    payload["explainer"] = shap.TreeExplainer(payload["model"])
    return payload


def _encode_input(raw: dict, payload: dict) -> np.ndarray:
    """Encode categorical fields using the saved LabelEncoders."""
    encoders  = payload["encoders"]
    cat_cols  = payload["cat_cols"]
    feat_cols = payload["feature_cols"]

    row = dict(raw)
    for col in cat_cols:
        le = encoders[col]
        val = str(row.get(col, ""))
        if val in le.classes_:
            row[col] = int(le.transform([val])[0])
        else:
            # Unknown category — use most common class (index 0)
            row[col] = 0

    return np.array([[row[f] for f in feat_cols]], dtype=float)


def _risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def _impact_label(shap_val: float) -> str:
    abs_val = abs(shap_val)
    if abs_val >= 0.15:
        return "high"
    if abs_val >= 0.07:
        return "medium"
    return "low"


def score_transaction(raw: dict) -> dict:
    """
    Score a single transaction.

    Returns a dict with:
        fraud_probability, risk_level, flagged, explanation
    """
    payload   = load_model()
    feat_cols = payload["feature_cols"]
    X         = _encode_input(raw, payload)

    prob  = float(payload["model"].predict_proba(X)[0][1])
    level = _risk_level(prob)

    # SHAP explanation — RF returns array of shape (n_samples, n_features, n_classes)
    shap_vals = payload["explainer"].shap_values(X)
    sv = np.array(shap_vals)
    # Shape possibilities: (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
    if sv.ndim == 3 and sv.shape[0] == 2:
        # (n_classes, n_samples, n_features) — take fraud class [1], first sample [0]
        fraud_shap = sv[1][0]
    elif sv.ndim == 3 and sv.shape[2] == 2:
        # (n_samples, n_features, n_classes) — take first sample [0], fraud class [1]
        fraud_shap = sv[0, :, 1]
    elif sv.ndim == 2:
        fraud_shap = sv[0]
    else:
        fraud_shap = sv.flatten()[:len(feat_cols)]

    # Build top factors (top 4 by absolute SHAP value)
    pairs = sorted(
        zip(feat_cols, fraud_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:4]

    factors = []
    for feat, sv in pairs:
        raw_val = raw.get(feat, "")
        desc    = FEATURE_DESCRIPTIONS.get(feat, feat)
        detail  = f"{desc}: {raw_val}"
        if sv > 0:
            detail += " (increases fraud risk)"
        else:
            detail += " (reduces fraud risk)"
        factors.append({
            "feature":     feat,
            "description": detail,
            "impact":      _impact_label(sv),
        })

    model_name = payload["meta"].get("best_model", "ML Model") + " v1"

    return {
        "fraud_probability": round(prob, 4),
        "risk_level":        level,
        "flagged":           level in ("HIGH", "MEDIUM"),
        "explanation": {
            "top_factors": factors,
            "model":       model_name,
        },
    }


def get_model_auc() -> float:
    return load_model()["meta"]["auc_roc"]


def get_model_comparison() -> dict:
    """Return full model comparison metadata from training run."""
    return load_model()["meta"]
