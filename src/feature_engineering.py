"""
feature_engineering.py
Builds the feature matrix (X) and target vectors (y) used for ML training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering", "ActivityScore",
]


def build_features(df: pd.DataFrame):
    """
    Returns X (feature matrix) and two target arrays:
      - y_gpa        : continuous GPA
      - y_grade      : GradeClass integer (0–4)
      - y_risk       : encoded RiskLevel (Low/Medium/High)
    """
    X = df[FEATURE_COLS].copy()

    y_gpa   = df["GPA"].values
    y_grade = df["GradeClass"].astype(int).values

    risk_enc = LabelEncoder()
    y_risk = risk_enc.fit_transform(df["RiskLevel"])

    return X, y_gpa, y_grade, y_risk, risk_enc


def scale_features(X_train, X_test=None):
    """Fit scaler on train; optionally transform a test set too."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    if X_test is not None:
        return X_train_sc, scaler.transform(X_test)
    return X_train_sc, scaler


def load_scaler():
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


def prepare_single_student(input_dict: dict) -> np.ndarray:
    """
    Given a dict of raw feature values from the UI form,
    return a scaled 2-D array ready for model.predict().
    """
    # ActivityScore derived
    input_dict["ActivityScore"] = sum([
        input_dict.get("Extracurricular", 0),
        input_dict.get("Sports", 0),
        input_dict.get("Music", 0),
        input_dict.get("Volunteering", 0),
    ])
    row = pd.DataFrame([{col: input_dict[col] for col in FEATURE_COLS}])
    scaler = load_scaler()
    return scaler.transform(row)
