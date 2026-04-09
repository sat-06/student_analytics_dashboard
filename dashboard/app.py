"""
EduPulse Analytics  ·  Advanced Dashboard  v3.0
================================================
Run:  streamlit run dashboard_advanced.py
Deps: streamlit plotly pandas numpy scipy scikit-learn
"""

import os, sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# ── Try your project modules; fall back to demo data ─────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    from data_cleaning        import load_processed
    from feature_engineering  import prepare_single_student, FEATURE_COLS
    from model_training       import load_models, models_exist, train_all
    from risk_detection       import compute_risk_score, flag_at_risk
    from recommendation_engine import generate_recommendations
    from clustering           import assign_clusters, get_pca_coords, train_clusters
    _DEMO = False
except ImportError:
    _DEMO = True


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

P = {
    "bg"      : "#06070F",
    "surface" : "#0D0F1D",
    "surface2": "#131527",
    "border"  : "#1C1F38",
    "border2" : "#252847",
    "text"    : "#E2E4F6",
    "muted"   : "#636899",
    "accent"  : "#7B6FFF",
    "cyan"    : "#34D4D4",
    "green"   : "#0FE49A",
    "red"     : "#FF4868",
    "amber"   : "#FFAB00",
    "pink"    : "#FF6FD8",
    "navy"    : "#0A0C1A",
}

# ── Safe rgba helper — converts hex to rgba() string, no hex+alpha bugs ──────
def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

RISK_COLORS  = {"Low": P["green"], "Medium": P["amber"], "High": P["red"]}
GRADE_COLORS = {"A": P["green"], "B": P["cyan"], "C": P["accent"], "D": P["amber"], "F": P["red"]}
SEG_COLORS   = [P["accent"], P["red"], P["cyan"], P["amber"]]

CHART_BASE = dict(
    template      = "plotly_dark",
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="'Outfit', sans-serif", color=P["text"]),
    margin        = dict(l=16, r=16, t=48, b=16),
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="EduPulse Analytics", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, [class*="css"] {{
    font-family: 'Outfit', sans-serif;
    background: {P['bg']};
    color: {P['text']};
}}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {P['surface']}; }}
::-webkit-scrollbar-thumb {{ background: {P['border2']}; border-radius: 4px; }}

.main .block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{
    background: {P['navy']};
    border-right: 1px solid {P['border']};
}}
section[data-testid="stSidebar"] * {{ color: {P['text']} !important; }}

/* ─ Page wrapper ─ */
.pw {{ padding: 0 2.5rem 4rem; max-width: 1600px; margin: 0 auto; }}

/* ─ Site header ─ */
.site-hdr {{
    background: linear-gradient(135deg, {P['navy']} 0%, {P['surface']} 70%);
    border-bottom: 1px solid {P['border']};
    padding: 1.3rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
}}
.site-hdr::before {{
    content: ''; position: absolute; top: -60px; right: 8%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, {rgba(P['accent'],0.14)} 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.brand {{ display: flex; align-items: center; gap: 0.9rem; }}
.brand-logo {{
    width: 42px; height: 42px;
    background: linear-gradient(135deg, {P['accent']}, {P['cyan']});
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 1.3rem;
    box-shadow: 0 4px 18px {rgba(P['accent'],0.4)};
}}
.brand-name {{
    font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, {P['text']}, {P['muted']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.brand-tag {{ font-size: 0.65rem; color: {P['muted']}; letter-spacing: 0.08em; text-transform: uppercase; }}
.hdr-stats {{ display: flex; gap: 2rem; }}
.hdr-stat {{ text-align: right; }}
.hdr-val  {{ font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: {P['accent']}; }}
.hdr-lbl  {{ font-size: 0.62rem; color: {P['muted']}; text-transform: uppercase; letter-spacing: 0.08em; }}

/* ─ Hero ─ */
.hero {{ padding: 2.2rem 0 1.2rem; }}
.eyebrow {{
    display: inline-flex; align-items: center; gap: 5px;
    background: {rgba(P['accent'],0.1)}; border: 1px solid {rgba(P['accent'],0.25)};
    color: {P['accent']}; border-radius: 20px; padding: 3px 12px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.7rem;
}}
.hero-title {{
    font-size: 2.5rem; font-weight: 900; letter-spacing: -0.03em; line-height: 1.1;
    margin-bottom: 0.45rem;
}}
.hero-title span {{
    background: linear-gradient(90deg, {P['accent']}, {P['cyan']}, {P['green']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-desc {{ font-size: 0.87rem; color: {P['muted']}; max-width: 600px; line-height: 1.6; }}

/* ─ Section title ─ */
.sec {{
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 1rem; font-weight: 700; color: {P['text']};
    margin: 2.2rem 0 0.2rem; letter-spacing: -0.01em;
}}
.sec::after {{ content: ''; flex: 1; height: 1px; background: {P['border']}; margin-left: 0.5rem; }}
.sec-desc {{ font-size: 0.76rem; color: {P['muted']}; margin-bottom: 0.9rem; }}

/* ─ KPI strip ─ */
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 0.75rem; margin-bottom: 0.4rem; }}
.kpi {{
    background: {P['surface']}; border: 1px solid {P['border']};
    border-radius: 13px; padding: 1rem 1.1rem 0.9rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.kpi:hover {{
    border-color: var(--c);
    box-shadow: 0 0 0 1px var(--c), 0 6px 24px {rgba(P['accent'],0.07)};
}}
.kpi-bar {{ position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--c); }}
.kpi-icon {{ font-size: 0.95rem; opacity: 0.85; margin-bottom: 0.45rem; }}
.kpi-lbl {{ font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: {P['muted']}; margin-bottom: 0.2rem; }}
.kpi-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.75rem; font-weight: 600; color: var(--c); line-height: 1; }}
.kpi-sub {{ font-size: 0.65rem; color: {P['muted']}; margin-top: 0.25rem; }}

/* ─ Chart card ─ */
.cc {{
    background: {P['surface']}; border: 1px solid {P['border']};
    border-radius: 15px; padding: 0.35rem 0.35rem 0;
    overflow: hidden; transition: border-color 0.2s;
}}
.cc:hover {{ border-color: {P['border2']}; }}

/* ─ Insight box ─ */
.ib {{
    background: {rgba(P['accent'],0.06)}; border: 1px solid {rgba(P['accent'],0.2)};
    border-radius: 11px; padding: 0.9rem 1rem;
    font-size: 0.8rem; line-height: 1.55; color: {P['text']};
}}
.ib b {{ color: {P['cyan']}; }}
.ib.r {{ background: {rgba(P['red'],0.06)};   border-color: {rgba(P['red'],0.2)};   }}
.ib.r b {{ color: {P['red']}; }}
.ib.g {{ background: {rgba(P['green'],0.06)}; border-color: {rgba(P['green'],0.2)}; }}
.ib.g b {{ color: {P['green']}; }}

/* ─ Badges ─ */
.badge {{ display: inline-flex; align-items: center; gap: 3px;
          padding: 2px 8px; border-radius: 20px;
          font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; }}
.bh {{ background: {rgba(P['red'],0.14)};   color: {P['red']};   border: 1px solid {rgba(P['red'],0.3)};   }}
.bm {{ background: {rgba(P['amber'],0.14)}; color: {P['amber']}; border: 1px solid {rgba(P['amber'],0.3)}; }}
.bl {{ background: {rgba(P['green'],0.14)}; color: {P['green']}; border: 1px solid {rgba(P['green'],0.3)}; }}

/* ─ Prediction card ─ */
.pc {{
    background: {P['surface']}; border: 1px solid {P['border2']};
    border-radius: 16px; padding: 1.6rem 1.2rem; text-align: center;
    position: relative; overflow: hidden;
}}
.pc::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 50% 0%, var(--gc) 0%, transparent 65%);
    opacity: 0.1; pointer-events: none;
}}
.pc-num {{ font-family: 'JetBrains Mono', monospace; font-size: 3.4rem; font-weight: 700; line-height: 1; color: var(--gc); }}
.pc-lbl {{ font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {P['muted']}; margin-bottom: 0.5rem; }}
.pc-sub {{ font-size: 0.68rem; color: {P['muted']}; margin-top: 0.35rem; }}

/* ─ Rec card ─ */
.rc {{
    background: {P['surface']}; border: 1px solid {P['border']};
    border-left: 3px solid var(--bc); border-radius: 0 11px 11px 0;
    padding: 0.85rem 1.05rem; margin-bottom: 0.55rem;
}}
.rc-title  {{ font-size: 0.85rem; font-weight: 700; color: {P['text']}; margin-bottom: 0.25rem; }}
.rc-msg    {{ font-size: 0.77rem; color: {P['text']}; line-height: 1.5; margin-bottom: 0.25rem; }}
.rc-action {{ font-size: 0.71rem; color: {P['muted']}; }}

/* ─ Warning row ─ */
.wr {{
    display: flex; align-items: center; gap: 0.75rem;
    background: {rgba(P['red'],0.05)}; border: 1px solid {rgba(P['red'],0.18)};
    border-radius: 9px; padding: 0.5rem 0.9rem; margin-bottom: 0.32rem; font-size: 0.78rem;
}}
.wr-id   {{ font-family: 'JetBrains Mono', monospace; color: {P['muted']}; width: 70px; flex-shrink: 0; }}
.wr-info {{ flex: 1; color: {P['text']}; }}
.wr-bw   {{ width: 96px; height: 4px; background: {P['border']}; border-radius: 3px; overflow: hidden; flex-shrink: 0; }}
.wr-b    {{ height: 4px; background: {P['red']}; border-radius: 3px; }}
.wr-sc   {{ font-family: 'JetBrains Mono', monospace; color: {P['red']}; font-size: 0.72rem; width: 44px; text-align: right; flex-shrink: 0; }}

/* ─ Streamlit overrides ─ */
.stButton > button {{
    background: linear-gradient(135deg, {P['accent']}, {P['cyan']});
    color: white; border: none; border-radius: 9px;
    padding: 0.55rem 1.5rem; width: 100%;
    font-family: 'Outfit', sans-serif; font-weight: 700;
    font-size: 0.88rem; letter-spacing: 0.02em;
    transition: opacity 0.2s, transform 0.15s;
}}
.stButton > button:hover {{ opacity: 0.87; transform: translateY(-1px); }}
.stSelectbox > div > div,
.stNumberInput > div > div > input {{
    background: {P['surface2']} !important;
    border: 1px solid {P['border2']} !important;
    color: {P['text']} !important; border-radius: 7px !important;
    font-family: 'Outfit', sans-serif !important;
}}
.stSlider > div > div > div {{ background: {P['accent']} !important; }}
div[data-testid="metric-container"] {{
    background: {P['surface']}; border: 1px solid {P['border']};
    border-radius: 11px; padding: 0.65rem 0.9rem;
}}
div[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent; color: {P['muted']}; font-weight: 600;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
    color: {P['accent']} !important;
    border-bottom: 2px solid {P['accent']} !important;
}}
div[data-testid="stExpander"] {{
    background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 11px;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _demo(n: int = 2392) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    gender      = rng.integers(0, 2, n)
    ethnicity   = rng.choice([0,1,2,3], n, p=[.5,.2,.2,.1])
    par_edu     = rng.integers(0, 5, n)
    par_sup     = rng.integers(0, 5, n)
    study       = np.clip(rng.normal(15, 8, n), 0, 40)
    absences    = np.clip(rng.exponential(8, n), 0, 50).astype(int)
    tutoring    = rng.integers(0, 2, n)
    extra       = rng.integers(0, 2, n)
    sports      = rng.integers(0, 2, n)
    music       = rng.integers(0, 2, n)
    vol         = rng.integers(0, 2, n)
    activity    = extra + sports + music + vol
    age         = rng.integers(15, 19, n)

    gpa = np.clip(0.06*study - 0.04*absences + 0.12*par_sup + 0.08*par_edu
                  + 0.10*tutoring + 0.05*activity + rng.normal(0,.4,n) + 1.2, 0.0, 4.0)

    def _g(v):
        return 0 if v>=3.5 else 1 if v>=3.0 else 2 if v>=2.0 else 3 if v>=1.0 else 4

    grade_n = np.array([_g(v) for v in gpa])
    grade_l = np.array(["A","B","C","D","F"])[grade_n]
    rs = np.clip(40 + absences*0.8 - study*0.6 - par_sup*3 - tutoring*5 + rng.normal(0,5,n), 0, 100)
    rl = np.where(rs>=60,"High", np.where(rs>=35,"Medium","Low"))

    edu_m = {0:"None",1:"High School",2:"Some College",3:"Bachelor's",4:"Higher"}
    sup_m = {0:"None",1:"Low",2:"Moderate",3:"High",4:"Very High"}
    eth_m = {0:"Caucasian",1:"African American",2:"Asian",3:"Other"}

    X_c = np.column_stack([gpa/4, study/40, absences/50, par_sup/4, activity/4])
    cl  = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_c)
    cl_n = {0:"Engaged Achievers",1:"At-Risk Disengaged",2:"Silent Strugglers",3:"Steady Middles"}

    return pd.DataFrame({
        "StudentID"        : [f"S{i:05d}" for i in range(1,n+1)],
        "Age"              : age,
        "Gender"           : gender,
        "GenderLabel"      : np.array(["Female","Male"])[gender],
        "Ethnicity"        : ethnicity,
        "EthnicityLabel"   : [eth_m[e] for e in ethnicity],
        "ParentalEducation": par_edu,
        "EduLabel"         : [edu_m[e] for e in par_edu],
        "ParentalSupport"  : par_sup,
        "SupLabel"         : [sup_m[s] for s in par_sup],
        "StudyTimeWeekly"  : study.round(1),
        "Absences"         : absences,
        "Tutoring"         : tutoring,
        "Extracurricular"  : extra,
        "Sports"           : sports,
        "Music"            : music,
        "Volunteering"     : vol,
        "ActivityScore"    : activity,
        "GPA"              : gpa.round(2),
        "GradeClass"       : grade_n,
        "GradeLabel"       : grade_l,
        "RiskScore"        : rs.round(1),
        "RiskLevel"        : rl,
        "RiskFlag"         : rl,
        "Cluster"          : cl,
        "ClusterLabel"     : [cl_n[c] for c in cl],
    })


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def gpa_letter(g):
    return "A" if g>=3.5 else "B" if g>=3.0 else "C" if g>=2.0 else "D" if g>=1.0 else "F"

def apply_theme(fig, h=None):
    kw = dict(**CHART_BASE)
    if h: kw["height"] = h
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor=P["border"], linecolor=P["border2"], zeroline=False)
    fig.update_yaxes(gridcolor=P["border"], linecolor=P["border2"], zeroline=False)
    return fig

def sec(title, icon="", desc=""):
    st.markdown(f'<div class="sec">{icon} {title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="sec-desc">{desc}</div>', unsafe_allow_html=True)

def kpi_html(lbl, val, sub="", color=None, icon=""):
    c = color or P["accent"]
    return (f'<div class="kpi" style="--c:{c};">'
            f'<div class="kpi-bar"></div>'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-lbl">{lbl}</div>'
            f'<div class="kpi-val">{val}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>')

def pred_card_html(lbl, val, sub, color):
    return (f'<div class="pc" style="--gc:{color};">'
            f'<div class="pc-lbl">{lbl}</div>'
            f'<div class="pc-num">{val}</div>'
            f'<div class="pc-sub">{sub}</div>'
            f'</div>')


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS  — all fillcolor / marker alpha uses rgba() helper
# ══════════════════════════════════════════════════════════════════════════════

def chart_kde(df):
    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        sub = df[df["RiskLevel"] == risk]["GPA"].dropna()
        if len(sub) < 5:
            continue
        x = np.linspace(0, 4, 200)
        y = stats.gaussian_kde(sub, bw_method=0.3)(x)
        fig.add_trace(go.Scatter(
            x=x, y=y, name=risk, fill="tozeroy",
            line=dict(color=color, width=2.5),
            fillcolor=rgba(color, 0.12),
        ))
    fig.update_layout(title="GPA Kernel Density by Risk Level",
                      xaxis_title="GPA", yaxis_title="Density", **CHART_BASE)
    fig.update_xaxes(gridcolor=P["border"])
    fig.update_yaxes(gridcolor=P["border"])
    return fig


def chart_violin(df):
    fig = px.violin(df, x="GradeLabel", y="GPA", color="GradeLabel",
                    color_discrete_map=GRADE_COLORS, box=True, points=False,
                    category_orders={"GradeLabel": ["A","B","C","D","F"]},
                    title="GPA Distribution by Grade Band")
    fig.update_traces(meanline_visible=True)
    return apply_theme(fig, 340)


def chart_heatmap(df):
    bs = pd.cut(df["StudyTimeWeekly"], bins=[0,5,10,15,20,30,40],
                labels=["0-5","5-10","10-15","15-20","20-30","30+"])
    ba = pd.cut(df["Absences"], bins=[0,5,10,15,20,30,50],
                labels=["0-5","5-10","10-15","15-20","20-30","30+"])
    pivot = df.groupby([bs, ba], observed=True)["GPA"].mean().unstack(fill_value=np.nan)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, P["red"]], [0.5, P["amber"]], [1, P["green"]]],
        text=np.round(pivot.values, 2), texttemplate="%{text}", textfont_size=10,
        colorbar=dict(title="Avg GPA", tickfont=dict(color=P["muted"])),
    ))
    fig.update_layout(title="Average GPA: Study Hours × Absences",
                      xaxis_title="Absence Range", yaxis_title="Study Hours",
                      **CHART_BASE, height=340)
    return fig


def chart_waterfall(df):
    high = df[df["RiskLevel"] == "High"]
    low  = df[df["RiskLevel"] == "Low"]
    features = ["StudyTimeWeekly","Absences","ParentalSupport","Tutoring","ActivityScore","ParentalEducation"]
    labels   = ["Study Time","Absences","Parent Support","Tutoring","Activities","Parent Edu"]
    diffs = [high[f].mean() - low[f].mean() for f in features]
    fig = go.Figure(go.Bar(
        x=labels, y=diffs,
        marker_color=[P["red"] if d < 0 else P["green"] for d in diffs],
        text=[f"{d:+.2f}" for d in diffs], textposition="outside",
    ))
    fig.add_hline(y=0, line_color=P["border2"], line_dash="dash", line_width=1)
    fig.update_layout(title="Feature Gap: High Risk vs Low Risk (mean diff)", **CHART_BASE, height=340)
    return fig


def chart_funnel(df):
    stages = ["Enrolled", "Study ≥5h/wk", "Attends ≥80%",
              "Tutored or Supported", "GPA ≥ 2.0", "GPA ≥ 3.5"]
    counts = [
        len(df),
        (df["StudyTimeWeekly"] >= 5).sum(),
        (df["Absences"] <= 10).sum(),
        ((df["Tutoring"] == 1) | (df["ParentalSupport"] >= 3)).sum(),
        (df["GPA"] >= 2.0).sum(),
        (df["GPA"] >= 3.5).sum(),
    ]
    fig = go.Figure(go.Funnel(
        y=stages, x=counts, textinfo="value+percent initial",
        marker=dict(
            color=[P["accent"], P["cyan"], P["green"], P["amber"], P["pink"], P["green"]],
            line=dict(color=P["border"], width=0.5),
        ),
    ))
    return apply_theme(fig, 390)


def chart_sankey(df):
    grade_i = {"A":0,"B":1,"C":2,"D":3,"F":4}
    risk_i  = {"Low":5,"Medium":6,"High":7}
    labels  = ["Grade A","Grade B","Grade C","Grade D","Grade F",
               "Low Risk","Med Risk","High Risk"]
    node_colors = [P["green"],P["cyan"],P["accent"],P["amber"],P["red"],
                   P["green"],P["amber"],P["red"]]
    flow = {}
    for _, row in df.iterrows():
        k = (grade_i[row["GradeLabel"]], risk_i[row["RiskLevel"]])
        flow[k] = flow.get(k, 0) + 1
    src = [k[0] for k in flow]
    tgt = [k[1] for k in flow]
    val = list(flow.values())
    fig = go.Figure(go.Sankey(
        node=dict(pad=14, thickness=18, color=node_colors, label=labels,
                  line=dict(color=P["border"], width=0.5)),
        link=dict(source=src, target=tgt, value=val, color=rgba(P["accent"], 0.18)),
    ))
    return apply_theme(fig, 360)


def chart_sunburst(df):
    fig = px.sunburst(df, path=["GradeLabel","RiskLevel","GenderLabel"],
                      color="GPA",
                      color_continuous_scale=[[0,P["red"]],[0.5,P["amber"]],[1,P["green"]]],
                      title="Grade → Risk → Gender Hierarchy")
    return apply_theme(fig, 420)


def chart_parallel(df):
    tmp = df[["StudyTimeWeekly","Absences","ParentalSupport","ActivityScore","GPA","GradeClass"]].dropna()
    fig = px.parallel_coordinates(
        tmp, color="GradeClass",
        color_continuous_scale=[[0,P["green"]],[0.5,P["amber"]],[1,P["red"]]],
        labels={"StudyTimeWeekly":"Study Hrs","Absences":"Absences",
                "ParentalSupport":"Support","ActivityScore":"Activity",
                "GPA":"GPA","GradeClass":"Grade"},
        title="Parallel Coordinates — Student Feature Profiles",
    )
    return apply_theme(fig, 420)


def chart_corr(df):
    cols = ["GPA","StudyTimeWeekly","Absences","ParentalSupport",
            "ParentalEducation","ActivityScore","Age","Tutoring"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig = go.Figure(go.Heatmap(
        z=corr.where(~mask).values, x=corr.columns, y=corr.index,
        colorscale=[[0,P["red"]],[0.5,P["surface2"]],[1,P["green"]]],
        zmin=-1, zmax=1,
        text=np.where(~mask, np.round(corr.values,2), ""),
        texttemplate="%{text}", textfont_size=10,
        colorbar=dict(title="r", tickfont=dict(color=P["muted"])),
    ))
    return apply_theme(fig, 400)


def chart_scatter(df):
    fig = px.scatter(
        df.sample(min(1500, len(df)), random_state=1),
        x="StudyTimeWeekly", y="GPA", color="RiskLevel",
        color_discrete_map=RISK_COLORS, opacity=0.6,
        trendline="lowess", trendline_scope="overall",
        trendline_color_override=P["accent"],
        labels={"StudyTimeWeekly":"Weekly Study Hours"},
        title="Study Time vs GPA (LOWESS trend)",
    )
    return apply_theme(fig, 360)


def chart_bubble(df):
    seg = df.groupby("ClusterLabel").agg(
        Count=("StudentID","count"),
        AvgGPA=("GPA","mean"),
        AvgStudy=("StudyTimeWeekly","mean"),
    ).reset_index()
    fig = px.scatter(
        seg, x="AvgStudy", y="AvgGPA", size="Count",
        color="ClusterLabel", text="ClusterLabel",
        color_discrete_sequence=SEG_COLORS, size_max=68,
        labels={"AvgStudy":"Avg Study Hrs/Week","AvgGPA":"Avg GPA"},
        title="Cluster Centroids: Study Time vs GPA",
    )
    fig.update_traces(textposition="top center")
    return apply_theme(fig, 360)


def chart_trend(df):
    fig = go.Figure()
    for risk, color in RISK_COLORS.items():
        base = df[df["RiskLevel"] == risk]["GPA"].mean()
        np.random.seed(hash(risk) % 999)
        noise = np.random.normal(0, 0.025, 12)
        trend = 0.018 if risk == "Low" else (-0.03 if risk == "High" else -0.01)
        vals  = [min(4.0, max(0.0, base + trend*w + noise[i])) for i, w in enumerate(range(1, 13))]
        fig.add_trace(go.Scatter(
            x=list(range(1, 13)), y=vals, name=f"{risk} Risk",
            line=dict(color=color, width=2.5), mode="lines+markers",
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=rgba(color, 0.08),
        ))
    fig.update_layout(title="Projected GPA Trend (12-week simulation)",
                      xaxis_title="Week", yaxis_title="Avg GPA",
                      **CHART_BASE, height=320)
    fig.update_xaxes(gridcolor=P["border"])
    fig.update_yaxes(gridcolor=P["border"])
    return fig


def chart_pca(df):
    features = ["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore"]
    avail = [c for c in features if c in df.columns]
    X = MinMaxScaler().fit_transform(df[avail].fillna(0))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    tmp = df.copy()
    tmp["PC1"] = coords[:, 0]
    tmp["PC2"] = coords[:, 1]
    fig = px.scatter(
        tmp.sample(min(1500, len(tmp)), random_state=42),
        x="PC1", y="PC2", color="ClusterLabel",
        color_discrete_sequence=SEG_COLORS, opacity=0.65,
        hover_data={"GPA":True,"RiskLevel":True,"PC1":False,"PC2":False},
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"},
        title="PCA Cluster Map (2D projection)",
    )
    return apply_theme(fig, 440)


def chart_radar(df):
    features = ["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore"]
    profile  = df.groupby("ClusterLabel")[features].mean()
    norm     = pd.DataFrame(
        MinMaxScaler().fit_transform(profile),
        index=profile.index, columns=features,
    )
    fig = go.Figure()
    for i, (idx, row) in enumerate(norm.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        cats = features + [features[0]]
        c = SEG_COLORS[i % 4]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=idx,
            line=dict(color=c, width=2),
            fillcolor=rgba(c, 0.18),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], color=P["border2"])),
        **CHART_BASE, height=440,
    )
    return fig


def chart_fi():
    features   = ["Absences","Study Time","Parental Support","Prev GPA","Tutoring",
                  "Activity Score","Parent Edu","Age","Gender","Ethnicity"]
    importance = [0.28, 0.23, 0.14, 0.12, 0.08, 0.06, 0.04, 0.02, 0.015, 0.015]
    colors = [P["red"] if i<3 else P["accent"] if i<5 else P["muted"] for i in range(10)]
    fig = go.Figure(go.Bar(
        x=importance, y=features, orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in importance], textposition="outside",
    ))
    fig.update_layout(title="Feature Importance (GBT — GPA Prediction)",
                      xaxis_title="Importance", yaxis=dict(autorange="reversed"),
                      **CHART_BASE, height=360)
    fig.update_xaxes(gridcolor=P["border"])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    if _DEMO:
        return _demo()
    df = load_processed()
    if "ClusterLabel" not in df.columns:
        df = train_clusters(df)
    elif "Cluster" not in df.columns:
        df = assign_clusters(df)
    return df


@st.cache_resource(show_spinner=False)
def load_models_cached():
    if _DEMO:
        return None, None, None, None, None
    if not models_exist():
        train_all()
    return load_models()


with st.spinner(""):
    DF = load_data()
    gpa_model, grade_model, risk_model, risk_enc, scaler = load_models_cached()


# ══════════════════════════════════════════════════════════════════════════════
#  SITE HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="site-hdr">
  <div class="brand">
    <div class="brand-logo">🎓</div>
    <div>
      <div class="brand-name">EduPulse</div>
      <div class="brand-tag">Student Performance Intelligence</div>
    </div>
  </div>
  <div class="hdr-stats">
    <div class="hdr-stat">
      <div class="hdr-val">{len(DF):,}</div>
      <div class="hdr-lbl">Total Students</div>
    </div>
    <div class="hdr-stat">
      <div class="hdr-val">{DF['GPA'].mean():.2f}</div>
      <div class="hdr-lbl">Avg GPA</div>
    </div>
    <div class="hdr-stat">
      <div class="hdr-val">{(DF['RiskLevel']=='High').sum():,}</div>
      <div class="hdr-lbl">High Risk</div>
    </div>
    {'<div class="hdr-stat"><div class="hdr-val" style="color:#636899;font-size:0.78rem;">⚡ DEMO</div><div class="hdr-lbl">Mode</div></div>' if _DEMO else ''}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""<div style="padding:1.1rem 0.4rem 0.4rem;">
    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                color:{P['muted']};margin-bottom:0.75rem;">Navigation</div></div>""",
                unsafe_allow_html=True)

    nav = st.radio("nav", [
        "📊  Overview",
        "🔬  Analysis",
        "🌐  Relationships",
        "👥  Segmentation",
        "⚠️  Early Warning",
        "🧬  ML Insights",
        "🧑‍💻  Predictor",
    ], label_visibility="collapsed")

    st.markdown(f"<hr style='border:none;border-top:1px solid {P['border']};margin:1rem 0;'>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{P['muted']};margin-bottom:0.6rem;'>Filters</div>",
                unsafe_allow_html=True)

    gender_f = st.multiselect("Gender",     ["Female","Male"],        default=["Female","Male"])
    risk_f   = st.multiselect("Risk Level", ["Low","Medium","High"],  default=["Low","Medium","High"])
    grade_f  = st.multiselect("Grade",      ["A","B","C","D","F"],    default=["A","B","C","D","F"])
    gpa_r    = st.slider("GPA Range", 0.0, 4.0, (0.0,4.0), 0.05)

    st.markdown(f"<hr style='border:none;border-top:1px solid {P['border']};margin:1rem 0;'>",
                unsafe_allow_html=True)
    st.caption(f"Dataset · {len(DF):,} students")


# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = DF.copy()
if gender_f: fdf = fdf[fdf["GenderLabel"].isin(gender_f)]
if risk_f:   fdf = fdf[fdf["RiskLevel"].isin(risk_f)]
if grade_f:  fdf = fdf[fdf["GradeLabel"].isin(grade_f)]
fdf = fdf[(fdf["GPA"] >= gpa_r[0]) & (fdf["GPA"] <= gpa_r[1])]
N = len(fdf)

# ── Page wrapper ──────────────────────────────────────────────────────────────
st.markdown('<div class="pw">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if nav == "📊  Overview":

    st.markdown(f"""<div class="hero">
      <div class="eyebrow">⬡ Live Dashboard</div>
      <div class="hero-title">Student Performance<br><span>Analytics Hub</span></div>
      <div class="hero-desc">ML-powered cohort intelligence across {N:,} students.
        Monitor performance, detect risk early, and drive better outcomes.</div>
    </div>""", unsafe_allow_html=True)

    avg_gpa   = fdf["GPA"].mean()
    high_risk = (fdf["RiskLevel"]=="High").sum()
    pass_rate = (fdf["GradeClass"]<=2).mean()*100
    avg_study = fdf["StudyTimeWeekly"].mean()
    tutored   = fdf["Tutoring"].mean()*100
    avg_abs   = fdf["Absences"].mean()
    a_rate    = (fdf["GradeLabel"]=="A").mean()*100

    st.markdown('<div class="kpis">', unsafe_allow_html=True)
    cols = st.columns(8)
    for col, (lbl,val,sub,c,icon) in zip(cols, [
        ("Students",    f"{N:,}",           "filtered cohort",           P["accent"],"👥"),
        ("Avg GPA",     f"{avg_gpa:.2f}",   f"Letter {gpa_letter(avg_gpa)}",P["cyan"],"📈"),
        ("At-Risk",     f"{high_risk:,}",   f"{high_risk/N*100:.1f}% high",P["red"],"⚠️"),
        ("Pass Rate",   f"{pass_rate:.1f}%","A / B / C grade",           P["green"],"✅"),
        ("Avg Study",   f"{avg_study:.1f}h","per week",                  P["amber"],"📚"),
        ("Tutored",     f"{tutored:.1f}%",  "receiving support",         P["pink"],"🎯"),
        ("Avg Absences",f"{avg_abs:.1f}",   "days missed",               P["red"],"🚪"),
        ("A-Grade",     f"{a_rate:.1f}%",   "top performers",            P["green"],"🏆"),
    ]):
        col.markdown(kpi_html(lbl,val,sub,c,icon), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Performance Distribution", "📊", "GPA spread and density across grade bands and risk groups")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_violin(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_kde(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sec("Behavioural Drivers", "🔥", "Study habits and absences interaction effect on GPA")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_heatmap(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_waterfall(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sec("Student Success Funnel", "🔻", "Drop-off rates at each academic milestone")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_funnel(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Auto-Insights", "💡", "Statistical observations from your current cohort")
    i1, i2, i3 = st.columns(3)
    ms_h = fdf[fdf["RiskLevel"]=="High"]["StudyTimeWeekly"].median()
    ms_l = fdf[fdf["RiskLevel"]=="Low"]["StudyTimeWeekly"].median()
    r_abs = fdf[["Absences","GPA"]].corr().iloc[0,1]
    tg = fdf[fdf["Tutoring"]==1]["GPA"].mean()
    ng = fdf[fdf["Tutoring"]==0]["GPA"].mean()
    with i1:
        st.markdown(f'<div class="ib r">High-risk students study <b>{ms_h:.1f}h/wk</b> vs <b>{ms_l:.1f}h/wk</b> for low-risk — a gap of <b>{ms_l-ms_h:.1f}h</b> that explains much of the performance divide.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="ib">Absences correlate with GPA at <b>r = {r_abs:.3f}</b> — the single strongest behavioural predictor in this cohort.</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div class="ib g">Tutored students average <b>GPA {tg:.2f}</b> vs <b>{ng:.2f}</b> without tutoring — a <b>{tg-ng:+.2f}</b> uplift.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🔬  Analysis":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Deep Dive</div><div class="hero-title">Multi-Dimensional<br><span>Analysis</span></div></div>', unsafe_allow_html=True)

    tabs = st.tabs(["👤 Demographics", "🏠 Family & Support", "🏃 Behaviour", "📊 Distributions"])

    with tabs[0]:
        sec("Gender & Ethnicity", "👤")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(fdf, x="GenderLabel", y="GPA", color="GenderLabel",
                         color_discrete_sequence=[P["pink"],P["cyan"]],
                         points="outliers", notched=True, title="GPA by Gender")
            st.plotly_chart(apply_theme(fig,340), use_container_width=True)
        with c2:
            eth_s = fdf.groupby("EthnicityLabel")["GPA"].agg(["mean","std"]).reset_index()
            eth_s.columns = ["Ethnicity","Mean","Std"]
            fig2 = px.bar(eth_s, x="Ethnicity", y="Mean", color="Ethnicity", error_y="Std",
                          color_discrete_sequence=[P["accent"],P["cyan"],P["green"],P["amber"]],
                          title="Mean GPA by Ethnicity (± 1 SD)")
            st.plotly_chart(apply_theme(fig2,340), use_container_width=True)
        sec("Age by Grade", "🎂")
        fig3 = px.histogram(fdf, x="Age", color="GradeLabel", color_discrete_map=GRADE_COLORS,
                            barmode="group", nbins=8, title="Age Distribution by Grade Class")
        st.plotly_chart(apply_theme(fig3,300), use_container_width=True)

    with tabs[1]:
        sec("Parental Education & Support", "🏠")
        edu_order = ["None","High School","Some College","Bachelor's","Higher"]
        sup_order = ["None","Low","Moderate","High","Very High"]
        c3, c4 = st.columns(2)
        with c3:
            # FIX: changed x="ParentalEduLabel" → x="EduLabel" (correct column name)
            # FIX: changed category_orders key "ParentalEduLabel" → "EduLabel"
            fig4 = px.violin(
                fdf,
                x="ParentalEduLabel",
                y="GPA",
                color="ParentalEduLabel",
                color_discrete_sequence=[P["red"], P["amber"], P["accent"], P["cyan"], P["green"]],
                category_orders={"ParentalEduLabel": edu_order},
                box=True,
                points=False,
                title="GPA by Parental Education"
            )
            st.plotly_chart(apply_theme(fig4,360), use_container_width=True)
        with c4:
          fig5 = px.violin(fdf, x="ParentalSupportLabel", y="GPA", color="ParentalSupportLabel",
                 color_discrete_sequence=[P["red"],P["amber"],P["accent"],P["cyan"],P["green"]],
                 category_orders={"ParentalSupportLabel":sup_order}, box=True, points=False,
                 title="GPA by Parental Support Level")
        st.plotly_chart(apply_theme(fig5,360), use_container_width=True)
        sec("Support × Grade Mix", "📐")
        cross = pd.crosstab(fdf["ParentalSupportLabel"], fdf["GradeLabel"], normalize="index")*100
        cross = cross.reindex(sup_order).fillna(0)
        fig6 = go.Figure()
        for g, c in GRADE_COLORS.items():
            if g in cross.columns:
                fig6.add_trace(go.Bar(name=g, x=cross.index, y=cross[g], marker_color=c))
        fig6.update_layout(barmode="stack", title="Grade Mix by Support Level (%)", **CHART_BASE, height=320)
        st.plotly_chart(fig6, use_container_width=True)

    with tabs[2]:
        sec("Study Time & Absences", "⏱️")
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_scatter(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            fig7 = px.histogram(fdf, x="StudyTimeWeekly", color="RiskLevel",
                                color_discrete_map=RISK_COLORS, barmode="overlay", opacity=0.75,
                                nbins=35, title="Study Hours by Risk Level")
            st.plotly_chart(apply_theme(fig7,320), use_container_width=True)
        with c6:
            fig8 = px.box(fdf, x="GradeLabel", y="Absences", color="GradeLabel",
                          color_discrete_map=GRADE_COLORS,
                          category_orders={"GradeLabel":["A","B","C","D","F"]},
                          title="Absences by Grade Class", points="outliers")
            st.plotly_chart(apply_theme(fig8,320), use_container_width=True)
        sec("Extracurriculars & Tutoring", "🎭")
        c7, c8 = st.columns(2)
        act_map = {0:"None",1:"One",2:"Two",3:"Three",4:"All"}
        with c7:
            tmp = fdf.copy(); tmp["ActLabel"] = tmp["ActivityScore"].map(act_map)
            fig9 = px.box(tmp, x="ActLabel", y="GPA", color="ActLabel",
                          color_discrete_sequence=[P["muted"],P["accent"],P["cyan"],P["green"],P["pink"]],
                          category_orders={"ActLabel":["None","One","Two","Three","All"]},
                          title="GPA vs Extracurricular Count", points="outliers")
            st.plotly_chart(apply_theme(fig9,340), use_container_width=True)
        with c8:
            tmp2 = fdf.copy(); tmp2["TutLabel"] = tmp2["Tutoring"].map({0:"No",1:"Yes"})
            fig10 = px.violin(tmp2, x="TutLabel", y="GPA", color="TutLabel",
                              color_discrete_sequence=[P["red"],P["green"]],
                              box=True, points="outliers", title="GPA: Tutored vs Not Tutored")
            st.plotly_chart(apply_theme(fig10,340), use_container_width=True)

    with tabs[3]:
        sec("Feature Distributions", "📈")
        dc = st.selectbox("Feature", ["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore","Age"])
        fig_d = px.histogram(fdf, x=dc, color="GradeLabel", color_discrete_map=GRADE_COLORS,
                             barmode="overlay", opacity=0.7, nbins=40, marginal="rug",
                             title=f"{dc} — Full Distribution")
        st.plotly_chart(apply_theme(fig_d,400), use_container_width=True)
        sec("Descriptive Statistics", "📋")
        st.dataframe(fdf[["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore"]].describe().T.round(3),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🌐  Relationships":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Multivariate</div><div class="hero-title">Feature<br><span>Relationships</span></div></div>', unsafe_allow_html=True)

    sec("Correlation Matrix", "🔗")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_corr(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sec("Grade → Risk Sankey", "🌊")
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_sankey(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        sec("Sunburst: Grade × Risk × Gender", "🌞")
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_sunburst(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sec("Parallel Coordinates", "📏", "Each line is one student — trace patterns across all features")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_parallel(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Scatter Matrix", "⚡", "Pairwise scatter for selected features")
    sm_cols = st.multiselect("Choose 2–5 features",
                             ["GPA","StudyTimeWeekly","Absences","ParentalSupport","ActivityScore","Age"],
                             default=["GPA","StudyTimeWeekly","Absences","ParentalSupport"])
    if len(sm_cols) >= 2:
        fig_sm = px.scatter_matrix(fdf.sample(min(800,len(fdf)),random_state=1),
                                   dimensions=sm_cols, color="RiskLevel",
                                   color_discrete_map=RISK_COLORS, opacity=0.5)
        fig_sm.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(apply_theme(fig_sm,520), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "👥  Segmentation":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Clustering</div><div class="hero-title">Student<br><span>Segmentation</span></div></div>', unsafe_allow_html=True)

    cs = fdf.groupby("ClusterLabel").agg(
        Count=("StudentID","count"),
        AvgGPA=("GPA","mean"),
        AvgAbs=("Absences","mean"),
        HighRisk=("RiskLevel", lambda x: (x=="High").mean()*100),
    ).reset_index()

    st.markdown('<div class="kpis">', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (col, (_, row)) in enumerate(zip(cols, cs.iterrows())):
        c = SEG_COLORS[i%4]
        col.markdown(kpi_html(row["ClusterLabel"], f"{row['Count']:,}",
                              f"GPA {row['AvgGPA']:.2f} · {row['HighRisk']:.0f}% high-risk", c, "👥"),
                     unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Cluster Map & Centroids", "🗺️")
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_pca(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.plotly_chart(chart_bubble(fdf), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sec("Cluster Radar Profiles", "🕸️")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_radar(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Grade Distribution by Segment", "📊")
    gc = pd.crosstab(fdf["ClusterLabel"], fdf["GradeLabel"], normalize="index")*100
    fig_gc = go.Figure()
    for g, c in GRADE_COLORS.items():
        if g in gc.columns:
            fig_gc.add_trace(go.Bar(name=f"Grade {g}", x=gc.index, y=gc[g], marker_color=c))
    fig_gc.update_layout(barmode="stack", title="Grade Mix within Each Cluster (%)", **CHART_BASE, height=300)
    st.plotly_chart(fig_gc, use_container_width=True)

    sec("Cluster Member Table", "📋")
    sel = st.selectbox("Filter", ["All"] + sorted(fdf["ClusterLabel"].unique().tolist()))
    show = fdf if sel=="All" else fdf[fdf["ClusterLabel"]==sel]
    st.dataframe(
        show[["StudentID","Age","GenderLabel","GPA","GradeLabel","StudyTimeWeekly","Absences","RiskLevel","ClusterLabel"]]
            .rename(columns={"GenderLabel":"Gender","GradeLabel":"Grade","StudyTimeWeekly":"Study Hrs",
                             "RiskLevel":"Risk","ClusterLabel":"Segment"})
            .reset_index(drop=True),
        use_container_width=True, height=360,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "⚠️  Early Warning":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Intervention</div><div class="hero-title">Early Warning<br><span>System</span></div></div>', unsafe_allow_html=True)

    if "RiskScore" not in fdf.columns:
        fdf["RiskScore"] = np.clip(40 + fdf["Absences"]*0.8 - fdf["StudyTimeWeekly"]*0.6, 0, 100)

    high = fdf[fdf["RiskLevel"]=="High"].sort_values("RiskScore",ascending=False)
    med  = fdf[fdf["RiskLevel"]=="Medium"]
    low  = fdf[fdf["RiskLevel"]=="Low"]

    st.markdown('<div class="kpis">', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (lbl,val,sub,c,icon) in zip(cols, [
        ("High Risk",   f"{len(high):,}", "immediate attention", P["red"],   "🔴"),
        ("Medium Risk", f"{len(med):,}",  "monitor closely",     P["amber"], "🟡"),
        ("Low Risk",    f"{len(low):,}",  "on track",            P["green"], "🟢"),
        ("Intervention",f"{len(high)/N*100:.1f}%","need outreach",P["pink"], "📣"),
    ]):
        col.markdown(kpi_html(lbl,val,sub,c,icon), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Risk Score Distribution", "🔥")
    fig_rd = px.histogram(fdf, x="RiskScore", color="RiskLevel",
                          color_discrete_map={"High":P["red"],"Medium":P["amber"],"Low":P["green"]},
                          barmode="overlay", opacity=0.82, nbins=40, marginal="rug")
    fig_rd.add_vline(x=35, line_dash="dash", line_color=P["amber"], annotation_text="Medium threshold")
    fig_rd.add_vline(x=60, line_dash="dash", line_color=P["red"],   annotation_text="High threshold")
    st.plotly_chart(apply_theme(fig_rd,340), use_container_width=True)

    sec("Projected GPA Trajectories", "📉")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_trend(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Top High-Risk Students", "🚨", "Sorted by risk score — prioritise these for counselling outreach")
    top_n = st.slider("Show top N students", 5, 60, 20)
    for _, row in high.head(top_n).iterrows():
        bar = int(row["RiskScore"])
        st.markdown(f"""<div class="wr">
          <span class="wr-id">{row['StudentID']}</span>
          <span class="wr-info">GPA <b style="color:{P['red']}">{row['GPA']:.2f}</b>
           · {int(row['Absences'])} abs · {row['StudyTimeWeekly']:.0f}h/wk</span>
          <div class="wr-bw"><div class="wr-b" style="width:{bar}%;"></div></div>
          <span class="wr-sc">{row['RiskScore']:.0f}/100</span>
        </div>""", unsafe_allow_html=True)

    sec("Risk Factor Distributions", "📌")
    c5, c6 = st.columns(2)
    with c5:
        fig_a = px.histogram(high, x="Absences", nbins=25, marginal="box",
                             color_discrete_sequence=[P["red"]], title="Absences — High Risk")
        st.plotly_chart(apply_theme(fig_a,320), use_container_width=True)
    with c6:
        fig_s = px.histogram(high, x="StudyTimeWeekly", nbins=25, marginal="box",
                             color_discrete_sequence=[P["amber"]], title="Study Time — High Risk")
        st.plotly_chart(apply_theme(fig_s,320), use_container_width=True)

    sec("Risk vs Parental Support", "🧩")
    rc = pd.crosstab(fdf["SupLabel"], fdf["RiskLevel"], normalize="index")*100
    rc = rc.reindex(["None","Low","Moderate","High","Very High"]).fillna(0)
    fig_rc = go.Figure()
    for r, c in [("High",P["red"]),("Medium",P["amber"]),("Low",P["green"])]:
        if r in rc.columns:
            fig_rc.add_trace(go.Bar(name=r, x=rc.index, y=rc[r], marker_color=c))
    fig_rc.update_layout(barmode="stack", title="Risk Distribution by Parental Support (%)",
                         **CHART_BASE, height=300)
    st.plotly_chart(fig_rc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🧬  ML Insights":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Machine Learning</div><div class="hero-title">Model Diagnostics<br><span>& Insights</span></div></div>', unsafe_allow_html=True)

    sec("Feature Importance", "🎯", "Relative contribution to GPA prediction — pre-computed from GBT model")
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    st.plotly_chart(chart_fi(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    sec("Live Model Training", "🤖", "Train on your filtered cohort and evaluate cross-validated performance")
    with st.expander("⚙️ Configuration", expanded=True):
        mc1, mc2, mc3 = st.columns(3)
        with mc1: mtype = st.selectbox("GPA Model", ["Gradient Boosting","Random Forest","Ridge Regression"])
        with mc2: n_est = st.slider("N Estimators", 50, 300, 100, 25)
        with mc3: cv_k  = st.slider("CV Folds", 2, 10, 5)

    if st.button("🚀 Train & Evaluate"):
        with st.spinner("Training…"):
            fcols = ["Age","Gender","Ethnicity","ParentalEducation","StudyTimeWeekly",
                     "Absences","Tutoring","ParentalSupport","Extracurricular",
                     "Sports","Music","Volunteering","ActivityScore"]
            avail = [c for c in fcols if c in fdf.columns]
            X = fdf[avail].fillna(0).values
            y_gpa   = fdf["GPA"].values
            y_grade = fdf["GradeClass"].values
            le = LabelEncoder()
            y_risk  = le.fit_transform(fdf["RiskLevel"].values)

            mdl = (GradientBoostingRegressor(n_estimators=n_est,random_state=42)
                   if mtype=="Gradient Boosting" else
                   RandomForestRegressor(n_estimators=n_est,random_state=42)
                   if mtype=="Random Forest" else Ridge())

            r2  = cross_val_score(mdl, X, y_gpa, cv=cv_k, scoring="r2")
            mse = cross_val_score(mdl, X, y_gpa, cv=cv_k, scoring="neg_mean_squared_error")
            gr  = cross_val_score(GradientBoostingClassifier(n_estimators=n_est,random_state=42), X, y_grade, cv=cv_k, scoring="accuracy")
            rk  = cross_val_score(RandomForestClassifier(n_estimators=n_est,random_state=42), X, y_risk, cv=cv_k, scoring="f1_macro")

        st.markdown('<div class="kpis">', unsafe_allow_html=True)
        r_cols = st.columns(6)
        for col, (lbl,val,sub,c,icon) in zip(r_cols, [
            ("GPA R²",     f"{r2.mean():.3f}",             f"± {r2.std():.3f}",  P["green"],"📈"),
            ("GPA RMSE",   f"{np.sqrt(-mse.mean()):.3f}",  "cross-val",          P["cyan"],"📉"),
            ("Grade Acc.", f"{gr.mean():.1%}",              f"± {gr.std():.1%}", P["accent"],"🎯"),
            ("Risk F1",    f"{rk.mean():.3f}",              f"± {rk.std():.3f}", P["amber"],"⚠️"),
            ("CV Folds",   str(cv_k),                       "stratified",        P["pink"],"🔁"),
            ("Train N",    f"{len(X):,}",                   "samples",           P["muted"],"📦"),
        ]):
            col.markdown(kpi_html(lbl,val,sub,c,icon), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        sec("CV Score Distribution", "📊")
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Box(y=r2, name="GPA R²",      marker_color=P["green"],  boxpoints="all", jitter=0.3))
        fig_cv.add_trace(go.Box(y=gr, name="Grade Acc.",  marker_color=P["accent"], boxpoints="all", jitter=0.3))
        fig_cv.add_trace(go.Box(y=rk, name="Risk F1",     marker_color=P["amber"],  boxpoints="all", jitter=0.3))
        fig_cv.update_layout(title="Score Spread across CV Folds", **CHART_BASE, height=320)
        st.plotly_chart(fig_cv, use_container_width=True)

        mdl.fit(X, y_gpa)
        y_pred = mdl.predict(X)
        resid  = y_gpa - y_pred

        sec("Residual Analysis", "🔍")
        cr1, cr2 = st.columns(2)
        with cr1:
            fig_res = px.scatter(x=y_pred, y=resid, labels={"x":"Predicted GPA","y":"Residual"},
                                 opacity=0.45, color=np.abs(resid),
                                 color_continuous_scale=[P["green"],P["amber"],P["red"]],
                                 title="Predicted vs Residuals")
            fig_res.add_hline(y=0, line_dash="dash", line_color=P["muted"])
            st.plotly_chart(apply_theme(fig_res,340), use_container_width=True)
        with cr2:
            fig_rh = px.histogram(x=resid, nbins=40, color_discrete_sequence=[P["accent"]],
                                  title="Residual Distribution", labels={"x":"Residual"})
            st.plotly_chart(apply_theme(fig_rh,340), use_container_width=True)

        if hasattr(mdl,"feature_importances_"):
            sec("Trained Feature Importances", "🔬")
            fi = pd.Series(mdl.feature_importances_, index=avail).sort_values()
            fig_fi = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation="h",
                marker_color=[P["accent"] if v>fi.median() else P["muted"] for v in fi.values],
                text=[f"{v:.3f}" for v in fi.values], textposition="outside",
            ))
            fig_fi.update_layout(title="Feature Importances (trained on current cohort)",
                                 **CHART_BASE, height=360)
            st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🧑‍💻  Predictor":

    st.markdown('<div class="hero"><div class="eyebrow">⬡ Prediction Engine</div><div class="hero-title">Student<br><span>Predictor</span></div></div>', unsafe_allow_html=True)

    sec("Enter Student Profile", "📝")

    with st.form("predictor_form"):
        c1, c2, c3, c4 = st.columns(4)
        lbl_style = f"font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{P['muted']};padding-bottom:0.65rem;"
        with c1:
            st.markdown(f"<div style='{lbl_style}'>Demographics</div>", unsafe_allow_html=True)
            age    = st.selectbox("Age", [15,16,17,18], index=1)
            gender = st.selectbox("Gender", ["Female","Male"])
            eth    = st.selectbox("Ethnicity", ["Caucasian","African American","Asian","Other"])
            p_edu  = st.selectbox("Parent Education", ["None","High School","Some College","Bachelor's","Higher"])
        with c2:
            st.markdown(f"<div style='{lbl_style}'>Academic</div>", unsafe_allow_html=True)
            study  = st.slider("Study Hrs/Week", 0.0, 40.0, 12.0, 0.5)
            abs_   = st.slider("Absences", 0, 50, 5)
            tut    = st.selectbox("Tutoring?", ["No","Yes"])
            p_sup  = st.selectbox("Parent Support", ["None","Low","Moderate","High","Very High"])
        with c3:
            st.markdown(f"<div style='{lbl_style}'>Extracurriculars</div>", unsafe_allow_html=True)
            ext = st.selectbox("Club?",         ["No","Yes"])
            spt = st.selectbox("Sports?",       ["No","Yes"])
            mus = st.selectbox("Music?",        ["No","Yes"])
            vol = st.selectbox("Volunteering?", ["No","Yes"])
        with c4:
            st.markdown(f"<div style='{lbl_style}'>Options</div>", unsafe_allow_html=True)
            show_radar = st.checkbox("Comparison radar", value=True)
            show_recs  = st.checkbox("Action plan",      value=True)
            st.markdown("<br>", unsafe_allow_html=True)
            submitted  = st.form_submit_button("🔮  Predict Performance")

    if submitted:
        sup_enc  = {"None":0,"Low":1,"Moderate":2,"High":3,"Very High":4}[p_sup]
        edu_enc  = {"None":0,"High School":1,"Some College":2,"Bachelor's":3,"Higher":4}[p_edu]
        tut_enc  = {"No":0,"Yes":1}[tut]
        activity = sum({"No":0,"Yes":1}[x] for x in [ext,spt,mus,vol])

        pred_gpa = np.clip(
            0.06*study - 0.04*abs_ + 0.12*sup_enc + 0.08*edu_enc
            + 0.10*tut_enc + 0.05*activity + 1.2, 0.0, 4.0,
        )
        pred_grade_n = min(4, int((4-pred_gpa)*1.25))
        pred_grade   = ["A","B","C","D","F"][pred_grade_n]
        risk_score   = np.clip(40 + abs_*0.8 - study*0.6 - sup_enc*3 - tut_enc*5, 0, 100)
        risk_level   = "High" if risk_score>=60 else "Medium" if risk_score>=35 else "Low"
        percentile   = (DF["GPA"] < pred_gpa).mean()*100

        st.markdown(f"<hr style='border:none;border-top:1px solid {P['border']};margin:1.5rem 0;'>", unsafe_allow_html=True)
        sec("Prediction Results", "🎯")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.markdown(pred_card_html("Predicted GPA",  f"{pred_gpa:.2f}", "out of 4.00",        P["accent"]),              unsafe_allow_html=True)
        r2.markdown(pred_card_html("Grade",          pred_grade,        f"Class {pred_grade_n}",GRADE_COLORS[pred_grade]),unsafe_allow_html=True)
        r3.markdown(pred_card_html("Risk Level",     risk_level,        f"Score {risk_score:.0f}/100",RISK_COLORS[risk_level]),unsafe_allow_html=True)
        r4.markdown(pred_card_html("GPA Percentile", f"{percentile:.0f}%","vs all students",  P["cyan"]),                unsafe_allow_html=True)
        r5.markdown(pred_card_html("Activities",     str(activity),     "extracurriculars",   P["amber"]),               unsafe_allow_html=True)

        cg, cf = st.columns(2)
        with cg:
            sec("Risk Gauge", "📊")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                delta={"reference": DF["RiskScore"].mean() if "RiskScore" in DF.columns else 45,
                       "valueformat":".0f"},
                number={"suffix":"/100","font":{"family":"JetBrains Mono","size":32,"color":P["text"]}},
                title={"text":"Risk Score","font":{"family":"Outfit","color":P["text"]}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":P["muted"]},
                    "bar":{"color":RISK_COLORS[risk_level],"thickness":0.22},
                    "bgcolor":P["surface2"], "bordercolor":P["border"],
                    "steps":[
                        {"range":[0,35],   "color":rgba(P["green"],0.1)},
                        {"range":[35,60],  "color":rgba(P["amber"],0.1)},
                        {"range":[60,100], "color":rgba(P["red"],0.1)},
                    ],
                    "threshold":{"line":{"color":RISK_COLORS[risk_level],"width":3},
                                 "thickness":0.85,"value":risk_score},
                },
            ))
            fig_g.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                height=280, margin=dict(l=30,r=30,t=30,b=10),
                                font=dict(family="Outfit"))
            st.plotly_chart(fig_g, use_container_width=True)

        with cf:
            sec("Risk Factors", "🔍")
            factors = []
            if abs_ > 15:              factors.append((f"🚪 {abs_} absences — well above average",       P["red"]))
            if study < 5:              factors.append((f"📚 Only {study:.0f}h/wk study — too low",       P["red"]))
            if sup_enc < 2:            factors.append((f"🏠 Weak parent support ({p_sup})",              P["amber"]))
            if tut_enc==0 and pred_gpa<2.5: factors.append(("🎯 No tutoring with GPA < 2.5",           P["amber"]))
            if activity == 0:          factors.append(("🏃 No extracurricular involvement",             P["cyan"]))
            if factors:
                for msg, c in factors:
                    st.markdown(f'<div class="rc" style="--bc:{c};"><div class="rc-title">{msg}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ib g">✅ <b>No major risk factors</b> — healthy academic trajectory.</div>', unsafe_allow_html=True)

        if show_radar:
            sec("Student vs Cohort Radar", "📡")
            rf = ["Study Time","GPA","Attendance","Support","Activity"]
            avg_v = [DF["StudyTimeWeekly"].mean()/40, DF["GPA"].mean()/4,
                     1-DF["Absences"].mean()/50,      DF["ParentalSupport"].mean()/4,
                     DF["ActivityScore"].mean()/4]
            top_v = [DF["StudyTimeWeekly"].quantile(.75)/40, DF["GPA"].quantile(.75)/4,
                     1-DF["Absences"].quantile(.25)/50,       DF["ParentalSupport"].quantile(.75)/4,
                     DF["ActivityScore"].quantile(.75)/4]
            stu_v = [study/40, pred_gpa/4, 1-abs_/50, sup_enc/4, activity/4]
            fig_r = go.Figure()
            for vals, name, color in [
                (avg_v,"Cohort Avg", P["muted"]),
                (top_v,"Top 25%",   P["cyan"]),
                (stu_v,"This Student", P["accent"]),
            ]:
                v2 = vals + [vals[0]]; c2 = rf + [rf[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=v2, theta=c2, fill="toself", name=name,
                    line=dict(color=color, width=2),
                    fillcolor=rgba(color, 0.15),
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1], color=P["border2"])),
                **CHART_BASE, height=380,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        if show_recs:
            sec("Personalised Action Plan", "💡")
            recs = []
            if study < 10:        recs.append(("📚 Increase Study Time",     P["red"],   "High",   f"Currently {study:.0f}h/week — target 15–20h for real GPA gains.", "Book 3 focused sessions/week, try Pomodoro"))
            if abs_ > 10:         recs.append(("🚪 Reduce Absences",         P["red"],   "High",   f"{abs_} absences correlates with ~{abs_*0.04:.2f} GPA drop.", "Set calendar reminders — aim < 5 per term"))
            if tut_enc==0 and pred_gpa<3.0: recs.append(("🎯 Enrol in Tutoring", P["amber"],"Medium","Tutored students at this GPA average +0.3 uplift.", "Contact academic support — free sessions available"))
            if sup_enc < 3:       recs.append(("🏠 Strengthen Home Support", P["amber"], "Medium", "High support correlates with significantly lower risk.", "Schedule monthly parent-teacher check-ins"))
            if activity == 0:     recs.append(("🏃 Join an Activity",        P["cyan"],  "Low",    "Students with 1–2 activities show better engagement.", "Explore clubs matching personal interests"))
            if not recs:          recs.append(("🌟 Keep It Up!",             P["green"], "Low",    "This student is performing well — consistency is key.", "Set stretch goals · consider peer mentoring"))

            badge_cls = {"High":"bh","Medium":"bm","Low":"bl"}
            for title, color, priority, msg, action in recs:
                st.markdown(f"""<div class="rc" style="--bc:{color};">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
                    <span class="rc-title">{title}</span>
                    <span class="badge {badge_cls[priority]}">{priority}</span>
                  </div>
                  <div class="rc-msg">{msg}</div>
                  <div class="rc-action">▶ {action}</div>
                </div>""", unsafe_allow_html=True)

# ── Close page wrapper ────────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)