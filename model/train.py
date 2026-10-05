"""
Model training script — compares multiple classifiers on realistic synthetic fraud data,
then saves the best-performing model as fraud_model.pkl.

Based on statistics from the public Kaggle Credit Card Fraud dataset
(ULB Machine Learning Group, 284,807 transactions, 0.172% fraud rate).

Models compared
---------------
  1. Logistic Regression  (fast baseline)
  2. Decision Tree        (interpretable baseline)
  3. Random Forest        (current production model)
  4. XGBoost              (gradient boosting challenger)
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    classification_report,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED       = 42
N_SAMPLES  = 50_000          # realistic size (< Kaggle's 284k but same proportions)
FRAUD_RATE = 0.00172 * 10    # ~1.72% — slightly higher for better class balance in demos

OUT_DIR = Path(__file__).parent
MODEL_PATH = OUT_DIR / "fraud_model.pkl"
META_PATH  = OUT_DIR / "model_meta.json"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Realistic data generation
#   Statistics are derived from the published Kaggle EDA literature so the
#   distributions are realistic rather than perfectly separable.
# ---------------------------------------------------------------------------

def _legit(n: int) -> pd.DataFrame:
    """Generate n legitimate transactions with realistic distributions."""
    hour          = rng.integers(0, 24, n)
    # Legitimate transactions peak 9-18; flatten with some night activity
    hour_bias     = np.where((hour >= 9) & (hour <= 18), 1.4, 1.0)

    amount        = rng.lognormal(mean=3.2, sigma=1.2, size=n)        # median ~24, skewed
    amount        = np.clip(amount, 0.5, 8000)
    foreign_tx    = rng.binomial(1, 0.08, n)
    velocity_1h   = rng.poisson(lam=1.2, size=n)
    amount_vs_avg = rng.lognormal(mean=0.05, sigma=0.35, size=n)      # near 1.0
    amount_vs_avg = np.clip(amount_vs_avg, 0.1, 8.0)
    distance_km   = rng.exponential(scale=80, size=n)
    device_change = rng.binomial(1, 0.04, n)

    merchant_category = rng.choice(
        ["grocery", "restaurant", "retail", "transport", "entertainment", "pharmacy"],
        size=n, p=[0.30, 0.22, 0.20, 0.12, 0.10, 0.06]
    )
    currency = rng.choice(["EUR", "USD", "GBP"], size=n, p=[0.75, 0.15, 0.10])
    country  = rng.choice(["IE", "GB", "US", "DE", "FR"], size=n, p=[0.70, 0.10, 0.08, 0.07, 0.05])

    return pd.DataFrame({
        "amount": amount, "hour": hour, "foreign_tx": foreign_tx,
        "merchant_category": merchant_category, "currency": currency,
        "country": country, "velocity_1h": velocity_1h,
        "amount_vs_avg": amount_vs_avg, "distance_km": distance_km,
        "device_change": device_change, "fraud": 0,
    })


def _fraud(n: int) -> pd.DataFrame:
    """Generate n fraudulent transactions — realistic overlap with legit."""
    # Fraud has realistic patterns but NOT perfectly separable
    hour_probs    = np.array([0.06,0.06,0.07,0.07,0.05,0.04,0.03,0.03,0.04,0.04,0.04,0.04,
                              0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.05,0.06])
    hour_probs   /= hour_probs.sum()   # normalize to exactly 1.0
    hour          = rng.choice(np.arange(24), size=n, p=hour_probs)

    # Fraud often small (card-testing) or large; bimodal
    small_fraud   = rng.lognormal(mean=1.5, sigma=0.8, size=n)   # ~4.5 median
    large_fraud   = rng.lognormal(mean=5.0, sigma=1.0, size=n)   # ~148 median
    small_mask    = rng.random(n) < 0.45
    amount        = np.where(small_mask, small_fraud, large_fraud)
    amount        = np.clip(amount, 0.5, 8000)

    foreign_tx    = rng.binomial(1, 0.55, n)   # more foreign
    velocity_1h   = rng.poisson(lam=3.8, size=n)
    amount_vs_avg = rng.lognormal(mean=1.1, sigma=0.9, size=n)
    amount_vs_avg = np.clip(amount_vs_avg, 0.1, 20.0)
    distance_km   = rng.exponential(scale=1200, size=n)          # much larger
    device_change = rng.binomial(1, 0.35, n)

    merchant_category = rng.choice(
        ["electronics", "jewelry", "online", "entertainment", "retail", "grocery"],
        size=n, p=[0.28, 0.18, 0.22, 0.12, 0.12, 0.08]
    )
    currency = rng.choice(
        ["USD", "EUR", "GBP", "NGN", "BRL"],
        size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    country = rng.choice(
        ["NG", "BR", "US", "IE", "GB", "CN"],
        size=n, p=[0.22, 0.18, 0.20, 0.15, 0.12, 0.13]
    )

    return pd.DataFrame({
        "amount": amount, "hour": hour, "foreign_tx": foreign_tx,
        "merchant_category": merchant_category, "currency": currency,
        "country": country, "velocity_1h": velocity_1h,
        "amount_vs_avg": amount_vs_avg, "distance_km": distance_km,
        "device_change": device_change, "fraud": 1,
    })


def generate_dataset() -> pd.DataFrame:
    n_fraud = int(N_SAMPLES * FRAUD_RATE)
    n_legit = N_SAMPLES - n_fraud
    df = pd.concat([_legit(n_legit), _fraud(n_fraud)], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Dataset: {len(df):,} transactions  |  fraud: {n_fraud:,} ({n_fraud/len(df)*100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CAT_COLS  = ["merchant_category", "currency", "country"]
FEAT_COLS = ["amount", "hour", "foreign_tx", "merchant_category",
             "currency", "country", "velocity_1h", "amount_vs_avg",
             "distance_km", "device_change"]


def encode(df: pd.DataFrame, encoders: dict | None = None):
    df = df.copy()
    fit_encoders = encoders is None
    if fit_encoders:
        encoders = {}
    for col in CAT_COLS:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: int(le.transform([v])[0]) if v in le.classes_ else 0
            )
    return df, encoders


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced",
            min_samples_leaf=20, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=int((1 - FRAUD_RATE) / FRAUD_RATE),
            eval_metric="logloss", use_label_encoder=False,
            random_state=SEED, n_jobs=-1
        )
    return models


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, name: str) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc    = roc_auc_score(y_test, proba)
    ap     = average_precision_score(y_test, proba)
    prec   = precision_score(y_test, pred, zero_division=0)
    rec    = recall_score(y_test, pred, zero_division=0)
    f1     = f1_score(y_test, pred, zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  AUC-ROC  : {auc:.4f}  |  Avg Precision: {ap:.4f}")
    print(f"  Precision: {prec:.4f} |  Recall:        {rec:.4f}  |  F1: {f1:.4f}")
    print(f"{'─'*50}")

    return {
        "model_name": name,
        "auc_roc":    round(auc,  4),
        "avg_prec":   round(ap,   4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1":         round(f1,   4),
    }


def train():
    print("=" * 60)
    print("  Fraud Detection — Model Comparison")
    print("=" * 60)

    # --- data ---
    df = generate_dataset()
    df_enc, encoders = encode(df)
    X = df_enc[FEAT_COLS].values
    y = df_enc["fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # --- train & compare ---
    models    = build_models()
    results   = []
    trained   = {}

    for name, clf in models.items():
        print(f"Training {name} …", end="", flush=True)
        clf.fit(X_train, y_train)
        print(" done")
        metrics = evaluate(clf, X_test, y_test, name)
        results.append(metrics)
        trained[name] = clf

    # --- pick best by AUC-ROC ---
    best_meta = max(results, key=lambda r: (r["auc_roc"], r["f1"]))
    best_name = best_meta["model_name"]
    best_clf  = trained[best_name]

    print(f"\n{'='*60}")
    print(f"  ✓ Best model: {best_name}  (AUC-ROC {best_meta['auc_roc']:.4f})")
    print(f"{'='*60}\n")

    # --- save best model ---
    payload = {
        "model":        best_clf,
        "encoders":     encoders,
        "cat_cols":     CAT_COLS,
        "feature_cols": FEAT_COLS,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    # --- save metadata (all comparison results + best) ---
    meta = {
        "best_model":     best_name,
        "auc_roc":        best_meta["auc_roc"],
        "avg_precision":  best_meta["avg_prec"],
        "precision":      best_meta["precision"],
        "recall":         best_meta["recall"],
        "f1":             best_meta["f1"],
        "n_features":     len(FEAT_COLS),
        "feature_cols":   FEAT_COLS,
        "train_samples":  int(len(X_train)),
        "fraud_rate_pct": round(FRAUD_RATE * 100, 3),
        "comparison":     results,            # all models' metrics
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved → {MODEL_PATH}")
    print(f"Meta  saved → {META_PATH}")
    print("\nModel comparison saved to model_meta.json → available at /api/v1/models/compare\n")

    return meta


if __name__ == "__main__":
    train()
