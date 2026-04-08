"""
data_cleaning.py
Handles loading, validating, and cleaning the raw student performance dataset.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "student_performance.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "students_clean.csv")

# ── Label maps ────────────────────────────────────────────────────────────────
GENDER_MAP          = {0: "Female", 1: "Male"}
ETHNICITY_MAP       = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
PARENTAL_EDU_MAP    = {0: "None", 1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Higher"}
PARENTAL_SUP_MAP    = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
GRADE_CLASS_MAP     = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all cleaning steps:
      - Drop duplicates
      - Clip numerical outliers to sensible ranges
      - Add human-readable label columns
      - Derive helper columns (RiskLevel, GPABand, ActivityScore)
    """
    df = df.copy()

    # ── Dedup ─────────────────────────────────────────────────────────────────
    df.drop_duplicates(subset="StudentID", inplace=True)

    # ── Clip numerical ranges ─────────────────────────────────────────────────
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].clip(0, 40)
    df["Absences"]        = df["Absences"].clip(0, 100)
    df["GPA"]             = df["GPA"].clip(0.0, 4.0)
    df["Age"]             = df["Age"].clip(15, 18)

    # ── Human-readable labels ─────────────────────────────────────────────────
    df["GenderLabel"]         = df["Gender"].map(GENDER_MAP)
    df["EthnicityLabel"]      = df["Ethnicity"].map(ETHNICITY_MAP)
    df["ParentalEduLabel"]    = df["ParentalEducation"].map(PARENTAL_EDU_MAP)
    df["ParentalSupportLabel"]= df["ParentalSupport"].map(PARENTAL_SUP_MAP)
    df["GradeLabel"]          = df["GradeClass"].map(GRADE_CLASS_MAP)

    # ── Derived features ──────────────────────────────────────────────────────
    # ActivityScore: number of extracurricular activities joined
    df["ActivityScore"] = df[["Extracurricular", "Sports", "Music", "Volunteering"]].sum(axis=1)

    # GPABand for readability
    df["GPABand"] = pd.cut(
        df["GPA"],
        bins=[0, 1.0, 2.0, 3.0, 4.0],
        labels=["Poor (0–1)", "Below Avg (1–2)", "Average (2–3)", "Excellent (3–4)"],
        include_lowest=True,
    )

    # RiskLevel based on GPA + absences
    def _risk(row):
        if row["GPA"] < 1.5 or row["Absences"] > 20:
            return "High"
        elif row["GPA"] < 2.5 or row["Absences"] > 10:
            return "Medium"
        else:
            return "Low"

    df["RiskLevel"] = df.apply(_risk, axis=1)

    return df


def save_processed(df: pd.DataFrame) -> None:
    """Persist the cleaned DataFrame to the processed folder."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)


def load_processed() -> pd.DataFrame:
    """Load the already-cleaned CSV; run pipeline if it doesn't exist yet."""
    if not os.path.exists(PROCESSED_PATH):
        raw = load_raw()
        cleaned = clean(raw)
        save_processed(cleaned)
        return cleaned
    return pd.read_csv(PROCESSED_PATH)


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    save_processed(cleaned)
    print(f"Cleaned dataset saved → {PROCESSED_PATH}")
    print(cleaned.shape)
    print(cleaned["RiskLevel"].value_counts())
