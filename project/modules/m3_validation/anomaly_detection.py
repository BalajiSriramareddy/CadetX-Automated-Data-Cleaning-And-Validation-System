"""
MODULE 3 · FILE 3 — ANOMALY DETECTION
=====================================
Finds rows that are unusual but do NOT break any explicit rule.

Two complementary methods:

  1. IQR outliers (per column, statistical)
     Flags values outside Q1 - 1.5*IQR .. Q3 + 1.5*IQR.
     Answers: "is this VALUE odd for this column?"

  2. Isolation Forest (multivariate, ML)
     Flags whole ROWS that are unusual across several columns at once.
     Answers: "is this COMBINATION of values odd?"
     e.g. tenure=1 month AND support_tickets=40 — neither value is extreme on
     its own, but together they are suspicious. IQR cannot see that; the
     Isolation Forest can.

Why Isolation Forest rather than a deep model? It is fast, needs no labelled
training data, handles mixed distributions, and is explainable: it isolates
points by random splits, and anomalies need fewer splits to isolate because
they sit apart from the dense mass of normal data.

Scope note: autoencoders / Local Outlier Factor are listed as future work.
One well-understood method beats three you cannot explain.

Design principle: DATASET-AGNOSTIC. Operates on whatever numeric columns exist.

Public entry point:  detect_anomalies(df, contamination=0.05) -> dict
"""
from __future__ import annotations

import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError:                      # keep the module usable without sklearn
    IsolationForest = None

MIN_ROWS = 20          # below this, "unusual" is not meaningful
MAX_EXAMPLES = 5
IQR_MULTIPLIER = 1.5


def iqr_outliers(df: pd.DataFrame) -> dict:
    """Per-column statistical outliers using the interquartile range."""
    results = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue                      # no spread; outliers are meaningless
        lo = q1 - IQR_MULTIPLIER * iqr
        hi = q3 + IQR_MULTIPLIER * iqr
        mask = (s < lo) | (s > hi)
        n = int(mask.sum())
        if n:
            results[col] = {
                "outliers": n,
                "outlier_pct": round(100 * n / len(s), 2),
                "bounds": {"lower": round(float(lo), 3), "upper": round(float(hi), 3)},
                "examples": [float(v) for v in s[mask].unique()[:MAX_EXAMPLES]],
            }
    return results


def isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """Multivariate row-level anomaly detection.

    contamination = the expected proportion of anomalies. It is an ASSUMPTION,
    not a discovery: set it too high and normal rows get flagged, too low and
    real anomalies are missed. 0.05 (5%) is a common starting point that should
    be tuned with domain input.
    """
    if IsolationForest is None:
        return {"method": "isolation_forest", "status": "sklearn not installed"}

    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] == 0 or len(numeric) < MIN_ROWS:
        return {"method": "isolation_forest",
                "status": f"skipped (needs >= {MIN_ROWS} rows and numeric columns)"}

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(numeric)          # -1 = anomaly, 1 = normal
    scores = model.score_samples(numeric)             # lower = more anomalous

    anomaly_mask = predictions == -1
    anomaly_idx = numeric.index[anomaly_mask].tolist()

    return {
        "method": "isolation_forest",
        "status": "ok",
        "contamination_assumed": contamination,
        "columns_used": list(numeric.columns),
        "rows_scored": int(len(numeric)),
        "anomalies_found": int(anomaly_mask.sum()),
        "anomaly_pct": round(100 * float(anomaly_mask.mean()), 2),
        "example_row_indices": anomaly_idx[:MAX_EXAMPLES],
        "most_anomalous_rows": (
            numeric.loc[pd.Series(scores, index=numeric.index)
                        .nsmallest(min(3, len(numeric))).index]
            .head(3).to_dict(orient="index")
        ),
    }


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """Run both anomaly methods and return one dict."""
    return {
        "iqr_outliers": iqr_outliers(df),
        "multivariate": isolation_forest_anomalies(df, contamination),
    }