"""
MODULE 3 — DATA VALIDATION
==========================
Check the cleaned dataset for errors, anomalies, and rule violations.

FILE CONTRACT
  Input :  data/processed/cleaned_data.csv
  Output:  outputs/validation_report.json

WHAT TO BUILD (Week 3)
  RULE-BASED  (own this fully — it's high-value and needs no ML):
    - format checks      (email / phone / postcode patterns via regex)
    - range checks       (numeric values within sensible bounds)
    - categorical checks (values in an allowed set)
  ANOMALY DETECTION (keep light — ONE method so you can talk about it):
    - Isolation Forest on numeric columns (~5 lines of sklearn)
  SCORE:
    - overall dataset health score + per-rule pass/fail counts

Scope note: the brief lists autoencoders and NLP classifiers here. Solo, you
do NOT need them. Rule-based + Isolation Forest satisfies the rubric and keeps
this analyst-focused. Mention the advanced options in docs as "future work".
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import write_json, CLEANED_DATA, VALIDATION_REPORT  # noqa: E402

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def check_email_column(series: pd.Series) -> dict:
    """Example rule check: how many values look like valid emails."""
    valid = series.dropna().astype(str).str.match(EMAIL_RE)
    return {
        "checked": int(valid.shape[0]),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
    }


def rule_based_checks(df: pd.DataFrame) -> dict:
    """TODO: build out format / range / categorical rules per column.

    Start with the email example above and expand. Define expected ranges and
    allowed categories based on what M1 profiling told you about the data.
    """
    results = {}
    for col in df.columns:
        if "email" in col.lower():
            results[col] = check_email_column(df[col])
    # TODO: add range checks, categorical-membership checks
    return results


def anomaly_detection(df: pd.DataFrame) -> dict:
    """TODO (Week 3): fit IsolationForest on numeric columns, count outliers.

    from sklearn.ensemble import IsolationForest
    num = df.select_dtypes("number").dropna()
    flags = IsolationForest(contamination=0.05, random_state=42).fit_predict(num)
    return {"anomalies": int((flags == -1).sum())}
    """
    return {"anomalies": None, "note": "not yet implemented"}


def health_score(rules: dict, anomalies: dict) -> float:
    """TODO: combine rule pass-rate + anomaly rate into one 0-100 score."""
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Module 3 — Validation")
    parser.add_argument("--input", default=str(CLEANED_DATA))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rules = rule_based_checks(df)
    anomalies = anomaly_detection(df)
    write_json(VALIDATION_REPORT, {
        "rule_based": rules,
        "anomaly_detection": anomalies,
        "health_score": health_score(rules, anomalies),
    })
    print("[M3] Done.")


if __name__ == "__main__":
    main()
