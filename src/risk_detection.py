"""
risk_detection.py
Classifies individual students or a full DataFrame into Low / Medium / High risk.
"""

import numpy as np
import pandas as pd


GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}


def compute_risk_score(gpa: float, absences: int, study_time: float,
                        parental_support: int, tutoring: int) -> dict:
    """
    Rule-based risk scoring that complements the ML model.
    Returns a dict with 'score' (0–100), 'level', and 'factors'.
    """
    score = 0
    factors = []

    # GPA contribution (0–40 pts)
    gpa_pts = max(0, 40 - (gpa / 4.0) * 40)
    score  += gpa_pts
    if gpa < 1.5:
        factors.append("Very low GPA")
    elif gpa < 2.5:
        factors.append("Below average GPA")

    # Absences contribution (0–30 pts)
    abs_pts = min(30, absences * 1.2)
    score  += abs_pts
    if absences > 20:
        factors.append("Excessive absences")
    elif absences > 10:
        factors.append("Frequent absences")

    # Study time contribution (0–15 pts)
    if study_time < 5:
        score += 15
        factors.append("Very low study time")
    elif study_time < 10:
        score += 7

    # Parental support (0–10 pts)
    if parental_support == 0:
        score += 10
        factors.append("No parental support")
    elif parental_support == 1:
        score += 5

    # No tutoring (0–5 pts)
    if tutoring == 0:
        score += 5

    score = min(100, round(score))

    if score >= 60:
        level = "High"
    elif score >= 35:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level, "factors": factors}


def flag_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RiskScore and RiskFlag columns to an existing cleaned DataFrame.
    """
    results = df.apply(
        lambda r: compute_risk_score(
            r["GPA"], r["Absences"], r["StudyTimeWeekly"],
            r["ParentalSupport"], r["Tutoring"]
        ),
        axis=1,
        result_type="expand",
    )
    df = df.copy()
    df["RiskScore"] = results["score"]
    df["RiskFlag"]  = results["level"]
    return df


def get_early_warning_students(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N highest-risk students."""
    flagged = flag_at_risk(df)
    return (
        flagged[flagged["RiskFlag"] == "High"]
        .sort_values("RiskScore", ascending=False)
        .head(top_n)
    )
