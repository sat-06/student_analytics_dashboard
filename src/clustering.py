"""
clustering.py
Segments students into behavioural clusters using KMeans.
"""

import os, joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CLUSTER_FEATURES = [
    "StudyTimeWeekly", "Absences", "GPA",
    "ParentalSupport", "ActivityScore", "Tutoring",
]

CLUSTER_LABELS = {
    0: "🌟 High Achievers",
    1: "⚠️ At-Risk Students",
    2: "📘 Average Performers",
    3: "🔄 Inconsistent Learners",
}


def train_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Fit KMeans on CLUSTER_FEATURES, save the model, and return the
    DataFrame augmented with Cluster and ClusterLabel columns.
    """
    X = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_sc)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(km,     os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "cluster_scaler.pkl"))

    df = df.copy()
    df["Cluster"]      = km.labels_
    df["ClusterLabel"] = df["Cluster"].map(CLUSTER_LABELS)
    return df


def load_cluster_model():
    km     = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.pkl"))
    return km, scaler


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Assign cluster labels to a DataFrame using saved model."""
    km, scaler = load_cluster_model()
    X  = df[CLUSTER_FEATURES].copy()
    Xs = scaler.transform(X)
    df = df.copy()
    df["Cluster"]      = km.predict(Xs)
    df["ClusterLabel"] = df["Cluster"].map(CLUSTER_LABELS)
    return df


def get_pca_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with PCA1, PCA2, Cluster, ClusterLabel for scatter plot.
    Fits PCA from scratch on CLUSTER_FEATURES.
    """
    _, scaler = load_cluster_model()
    X  = df[CLUSTER_FEATURES].copy()
    Xs = scaler.transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    result = df[["StudentID", "GPA", "Absences", "StudyTimeWeekly", "ClusterLabel"]].copy()
    result["PCA1"] = coords[:, 0]
    result["PCA2"] = coords[:, 1]
    return result
