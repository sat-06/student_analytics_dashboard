"""
recommendation_engine.py
Generates personalised, actionable study recommendations for a student
based on their profile values.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Recommendation:
    category: str          # e.g. "Study Habits"
    priority: str          # "High" | "Medium" | "Low"
    message: str
    action: str            # concrete next step
    icon: str = "💡"


def generate_recommendations(
    gpa: float,
    study_time: float,
    absences: int,
    tutoring: int,
    parental_support: int,
    activity_score: int,
    predicted_grade: int,
) -> List[Recommendation]:
    """
    Returns a ranked list of Recommendation objects for a single student.
    """
    recs: List[Recommendation] = []

    # ── Study time ────────────────────────────────────────────────────────────
    if study_time < 5:
        recs.append(Recommendation(
            category="Study Habits",
            priority="High",
            icon="📚",
            message="Your weekly study time is critically low (under 5 hrs).",
            action="Block at least 1 hour of focused study every day. Use the Pomodoro technique: 25 min study, 5 min break.",
        ))
    elif study_time < 10:
        recs.append(Recommendation(
            category="Study Habits",
            priority="Medium",
            icon="📚",
            message="Study time is below average. Aim for 10–15 hrs/week.",
            action="Add 2 extra study sessions per week. Focus on your weakest subjects first.",
        ))

    # ── Absences ─────────────────────────────────────────────────────────────
    if absences > 20:
        recs.append(Recommendation(
            category="Attendance",
            priority="High",
            icon="🏫",
            message=f"You have {absences} absences — this is severely impacting your GPA.",
            action="Meet with your counselor this week. Create an attendance contract and set daily alarm reminders.",
        ))
    elif absences > 10:
        recs.append(Recommendation(
            category="Attendance",
            priority="Medium",
            icon="🏫",
            message=f"You have {absences} absences. Try to keep it under 5 per semester.",
            action="Review which days/periods you miss most. Address any specific barriers (transport, motivation, etc.).",
        ))

    # ── Tutoring ─────────────────────────────────────────────────────────────
    if tutoring == 0 and gpa < 2.5:
        recs.append(Recommendation(
            category="Academic Support",
            priority="High",
            icon="🎓",
            message="You're not receiving tutoring despite a below-average GPA.",
            action="Enroll in tutoring sessions this week — ask your teacher or check school resources.",
        ))
    elif tutoring == 0 and gpa < 3.0:
        recs.append(Recommendation(
            category="Academic Support",
            priority="Medium",
            icon="🎓",
            message="Tutoring could push your GPA above 3.0.",
            action="Look into peer tutoring or free online platforms (Khan Academy, Coursera).",
        ))

    # ── Parental support ──────────────────────────────────────────────────────
    if parental_support <= 1:
        recs.append(Recommendation(
            category="Support System",
            priority="Medium",
            icon="🤝",
            message="Low parental support detected. Building your own support network matters.",
            action="Join a study group or find a mentor at school. Accountability with peers greatly boosts performance.",
        ))

    # ── Extracurricular balance ───────────────────────────────────────────────
    if activity_score == 0:
        recs.append(Recommendation(
            category="Extracurricular",
            priority="Low",
            icon="⚽",
            message="No extracurricular activities. Balanced engagement improves focus and well-being.",
            action="Try one club or sport that interests you — even 1 activity shows positive academic correlation.",
        ))
    elif activity_score >= 3 and gpa < 2.0:
        recs.append(Recommendation(
            category="Extracurricular",
            priority="Medium",
            icon="⚖️",
            message="High activity count with a low GPA suggests possible over-commitment.",
            action="Consider pausing one activity this semester to focus on academics.",
        ))

    # ── GPA-specific push ─────────────────────────────────────────────────────
    if gpa >= 3.5:
        recs.append(Recommendation(
            category="Advanced Growth",
            priority="Low",
            icon="🚀",
            message="Excellent GPA! You're on the honour roll track.",
            action="Consider AP classes, competitions, or research projects to strengthen your college profile.",
        ))
    elif gpa >= 3.0:
        recs.append(Recommendation(
            category="Improvement",
            priority="Low",
            icon="📈",
            message="Good standing. A small push in study time could get you to 3.5+.",
            action="Identify your lowest-graded subject and dedicate 30 extra minutes per day to it.",
        ))

    # Sort: High → Medium → Low
    order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: order[r.priority])

    return recs
