"""
model_training.py
Trains GPA regression + GradeClass classification models and saves them.
"""

import os, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Local imports (works when run from project root OR via Streamlit)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import load_processed
from feature_engineering import build_features, scale_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_all():
    """Full training pipeline. Returns metric dict."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_processed()
    X, y_gpa, y_grade, y_risk, risk_enc = build_features(df)

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, yg_train, yg_test, ygr_train, ygr_test, yr_train, yr_test = \
        train_test_split(X, y_gpa, y_grade, y_risk, test_size=0.2, random_state=42)

    X_train_sc, X_test_sc = scale_features(X_train, X_test)

    # ── GPA Regressor ─────────────────────────────────────────────────────────
    gpa_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gpa_model.fit(X_train_sc, yg_train)
    gpa_preds = gpa_model.predict(X_test_sc)
    gpa_mae   = mean_absolute_error(yg_test, gpa_preds)

    joblib.dump(gpa_model, os.path.join(MODELS_DIR, "gpa_regressor.pkl"))

    # ── GradeClass Classifier ─────────────────────────────────────────────────
    grade_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    grade_model.fit(X_train_sc, ygr_train)
    grade_preds = grade_model.predict(X_test_sc)
    grade_acc   = accuracy_score(ygr_test, grade_preds)

    joblib.dump(grade_model,  os.path.join(MODELS_DIR, "grade_classifier.pkl"))
    joblib.dump(risk_enc,     os.path.join(MODELS_DIR, "risk_encoder.pkl"))

    # ── Risk Classifier ───────────────────────────────────────────────────────
    risk_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    risk_model.fit(X_train_sc, yr_train)
    risk_preds = risk_model.predict(X_test_sc)
    risk_acc   = accuracy_score(yr_test, risk_preds)

    joblib.dump(risk_model, os.path.join(MODELS_DIR, "risk_classifier.pkl"))

    metrics = {
        "gpa_mae":   round(gpa_mae, 4),
        "grade_acc": round(grade_acc * 100, 2),
        "risk_acc":  round(risk_acc * 100, 2),
    }
    print("✅ Models saved.")
    print(f"   GPA MAE       : {metrics['gpa_mae']}")
    print(f"   Grade Acc     : {metrics['grade_acc']}%")
    print(f"   Risk Acc      : {metrics['risk_acc']}%")
    return metrics


def load_models():
    """Load all saved model artifacts. Returns (gpa_model, grade_model, risk_model, risk_enc, scaler)."""
    from feature_engineering import load_scaler
    gpa_model   = joblib.load(os.path.join(MODELS_DIR, "gpa_regressor.pkl"))
    grade_model = joblib.load(os.path.join(MODELS_DIR, "grade_classifier.pkl"))
    risk_model  = joblib.load(os.path.join(MODELS_DIR, "risk_classifier.pkl"))
    risk_enc    = joblib.load(os.path.join(MODELS_DIR, "risk_encoder.pkl"))
    scaler      = load_scaler()
    return gpa_model, grade_model, risk_model, risk_enc, scaler


def models_exist():
    files = ["gpa_regressor.pkl", "grade_classifier.pkl", "risk_classifier.pkl",
             "risk_encoder.pkl", "scaler.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in files)


if __name__ == "__main__":
    train_all()
