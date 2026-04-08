"""
utils.py
Shared helper functions used across the dashboard and src modules.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette (consistent across all charts) ────────────────────────────
PALETTE  = ["#6C63FF", "#48CAE4", "#F77F00", "#06D6A0", "#EF476F"]
RISK_COLORS = {"Low": "#06D6A0", "Medium": "#F77F00", "High": "#EF476F"}
GRADE_COLORS = {"A": "#06D6A0", "B": "#6C63FF", "C": "#48CAE4", "D": "#F77F00", "F": "#EF476F"}

GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
RISK_ORDER = ["Low", "Medium", "High"]


# ── Formatting helpers ────────────────────────────────────────────────────────

def gpa_to_letter(gpa: float) -> str:
    if gpa >= 3.5: return "A"
    if gpa >= 3.0: return "B"
    if gpa >= 2.0: return "C"
    if gpa >= 1.0: return "D"
    return "F"


def risk_badge_html(level: str) -> str:
    colors = {"Low": "#06D6A0", "Medium": "#F77F00", "High": "#EF476F"}
    c = colors.get(level, "#aaa")
    return (
        f'<span style="background:{c};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85rem;">{level} Risk</span>'
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_gpa_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="GPA", nbins=40, color_discrete_sequence=["#6C63FF"],
        title="GPA Distribution",
        labels={"GPA": "GPA Score"},
        marginal="box",
    )
    fig.update_layout(**_base_layout())
    return fig


def chart_grade_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["GradeLabel"].value_counts().reindex(["A","B","C","D","F"])
    fig = px.bar(
        x=counts.index, y=counts.values,
        color=counts.index,
        color_discrete_map=GRADE_COLORS,
        title="Grade Class Distribution",
        labels={"x": "Grade", "y": "Students"},
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    return fig


def chart_risk_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["RiskLevel"].value_counts().reindex(RISK_ORDER)
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[RISK_COLORS[r] for r in counts.index],
        hole=0.55,
        textinfo="label+percent",
    ))
    fig.update_layout(title="Risk Level Breakdown", **_base_layout())
    return fig


def chart_study_vs_gpa(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="StudyTimeWeekly", y="GPA",
        color="RiskLevel",
        color_discrete_map=RISK_COLORS,
        opacity=0.65,
        title="Study Time vs GPA",
        labels={"StudyTimeWeekly": "Weekly Study Hours", "GPA": "GPA"},
        trendline="ols",
        hover_data=["Absences", "GradeLabel"],
    )
    fig.update_layout(**_base_layout())
    return fig


def chart_absences_vs_gpa(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="Absences", y="GPA",
        color="RiskLevel",
        color_discrete_map=RISK_COLORS,
        opacity=0.65,
        title="Absences vs GPA",
        labels={"Absences": "Number of Absences", "GPA": "GPA"},
        trendline="ols",
    )
    fig.update_layout(**_base_layout())
    return fig


def chart_parental_support_gpa(df: pd.DataFrame) -> go.Figure:
    mapping = {0:"None",1:"Low",2:"Moderate",3:"High",4:"Very High"}
    tmp = df.copy()
    tmp["SupportLabel"] = tmp["ParentalSupport"].map(mapping)
    order = ["None","Low","Moderate","High","Very High"]
    fig = px.box(
        tmp, x="SupportLabel", y="GPA",
        color="SupportLabel",
        color_discrete_sequence=PALETTE,
        category_orders={"SupportLabel": order},
        title="Parental Support vs GPA",
        labels={"SupportLabel": "Parental Support Level"},
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    return fig


def chart_activity_vs_gpa(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="ActivityScore", y="GPA",
        color="ActivityScore",
        color_discrete_sequence=PALETTE,
        title="Extracurricular Activity Count vs GPA",
        labels={"ActivityScore": "Number of Activities"},
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ["GPA","StudyTimeWeekly","Absences","ParentalSupport",
                "ActivityScore","Age","Tutoring","ParentalEducation"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Feature Correlation Heatmap", **_base_layout())
    return fig


def chart_gender_gpa(df: pd.DataFrame) -> go.Figure:
    tmp = df.copy()
    tmp["Gender"] = tmp["GenderLabel"]
    fig = px.violin(
        tmp, x="Gender", y="GPA", color="Gender",
        color_discrete_sequence=["#6C63FF","#48CAE4"],
        box=True, points="outliers",
        title="GPA Distribution by Gender",
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    return fig


def chart_ethnicity_gpa(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="EthnicityLabel", y="GPA",
        color="EthnicityLabel",
        color_discrete_sequence=PALETTE,
        title="GPA by Ethnicity",
        labels={"EthnicityLabel": "Ethnicity"},
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    return fig


def chart_cluster_scatter(pca_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        pca_df, x="PCA1", y="PCA2",
        color="ClusterLabel",
        color_discrete_sequence=PALETTE,
        hover_data=["StudentID","GPA","Absences","StudyTimeWeekly"],
        title="Student Segments (PCA View)",
        labels={"PCA1": "Component 1", "PCA2": "Component 2"},
        opacity=0.7,
    )
    fig.update_layout(**_base_layout())
    return fig


def chart_cluster_profile(df: pd.DataFrame) -> go.Figure:
    """Radar / bar chart of cluster mean profiles."""
    features = ["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore"]
    profile  = df.groupby("ClusterLabel")[features].mean().reset_index()
    fig = go.Figure()
    for i, row in profile.iterrows():
        fig.add_trace(go.Bar(
            name=row["ClusterLabel"],
            x=features,
            y=row[features].values,
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(
        barmode="group",
        title="Cluster Profiles (Mean Feature Values)",
        **_base_layout(),
    )
    return fig


# ── Internal layout helper ────────────────────────────────────────────────────

def _base_layout() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", color="#E0E0E0"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
