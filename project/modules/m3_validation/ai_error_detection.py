"""
MODULE 3 · AI-BASED ERROR DETECTION
====================================
Three checks that go beyond simple rule violations:

  1. Impossible values     -> physically/logically impossible, e.g. negative
                              age, a percentage over 100, a date in the future
                              for something that must have already happened.
  2. Suspicious patterns    -> statistical fingerprints of fabricated or
                              placeholder data (e.g. suspiciously round
                              numbers, repeated identical values, sequences
                              that are too perfectly incrementing).
  3. Data drift             -> a REFERENCE dataset (e.g. last month) vs a
                              CURRENT one: have the distributions shifted
                              enough that a model or dashboard trained on the
                              old data would now be misleading?

WHY "AI-based" rather than plain rules: these checks reason about the SHAPE of
the data statistically (distributions, entropy, population comparison) rather
than fixed thresholds on a single column — that is what earns the "AI-based"
framing the brief uses, as distinct from the simple format/range rules in
rule_validation.py.

Public entry points:
  detect_impossible_values(df, report=None) -> dict
  detect_suspicious_patterns(df) -> dict
  detect_data_drift(reference_df, current_df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MAX_EXAMPLES = 5

# columns whose NAME suggests a logical upper/lower bound that data can't
# violate, independent of any business rules.yaml — these are physical/logical
# impossibilities, not business preferences.
IMPOSSIBLE_RULES = {
    "age": {"min": 0, "max": 130},
    "percentage": {"min": 0, "max": 100},
    "pct": {"min": 0, "max": 100},
    "rate": {"min": 0, "max": 100},
}


def detect_impossible_values(df: pd.DataFrame, report: dict | None = None) -> dict:
    """Flag values that are impossible regardless of business context —
    negative counts/ages, percentages over 100, and future dates in columns
    that describe something that must already have happened (e.g. an
    'install_date' or 'created_at' in the future)."""
    findings = {}
    meta = (report or {}).get("metadata", {})

    for col in df.columns:
        name = col.lower()
        series = df[col]

        # 1. name-based logical bounds (age, percentage, rate)
        for hint, bounds in IMPOSSIBLE_RULES.items():
            if hint in name:
                numeric = pd.to_numeric(series, errors="coerce")
                bad = (numeric < bounds["min"]) | (numeric > bounds["max"])
                n = int(bad.sum())
                if n:
                    findings[col] = {
                        "issue": f"values outside logically possible range "
                                f"[{bounds['min']}, {bounds['max']}]",
                        "count": n,
                        "examples": numeric[bad].unique()[:MAX_EXAMPLES].tolist(),
                    }
                break

        # 2. any count/quantity-like column: negative values are impossible
        semantic = meta.get(col, {}).get("semantic_type")
        if semantic == "numeric" or pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if ("count" in name or "quantity" in name or "tickets" in name
                    or "tenure" in name) and (numeric < 0).any():
                n = int((numeric < 0).sum())
                findings.setdefault(col, {
                    "issue": "negative values in a column that cannot be negative",
                    "count": n,
                    "examples": numeric[numeric < 0].unique()[:MAX_EXAMPLES].tolist(),
                })

        # 3. dates that describe a past event but are set in the future
        if any(h in name for h in ("date", "created", "install", "signup")) and \
                not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            future = parsed > pd.Timestamp.now()
            n = int(future.sum())
            if n:
                findings[col] = {
                    "issue": "dates in the future for an event that should be in the past",
                    "count": n,
                    "examples": [str(d.date()) for d in parsed[future].dropna().unique()[:MAX_EXAMPLES]],
                }

    return findings


def detect_suspicious_patterns(df: pd.DataFrame) -> dict:
    """Flag statistical fingerprints of fabricated, placeholder, or
    low-quality data — patterns real-world measurements rarely produce:
      - too many suspiciously round numbers (100, 200, 1000...)
      - one value repeated far more than chance would suggest
      - a numeric column that increments in an unnaturally perfect sequence
    """
    findings = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        issues = []

        # round-number fingerprint
        round_frac = (s % 10 == 0).mean()
        if round_frac > 0.4:
            issues.append(f"{round_frac:.0%} of values are suspiciously round "
                          f"(multiples of 10) — real measurements rarely cluster like this")

        # single dominant value (placeholder / default value left unfilled)
        top_freq = s.value_counts(normalize=True).iloc[0]
        if top_freq > 0.3 and s.nunique() > 5:
            top_val = s.value_counts().index[0]
            issues.append(f"value {top_val} appears in {top_freq:.0%} of rows — "
                          f"possible unfilled default/placeholder")

        # perfect arithmetic sequence (fabricated test data fingerprint)
        sorted_s = s.sort_values().reset_index(drop=True)
        if len(sorted_s) > 10:
            diffs = sorted_s.diff().dropna()
            if diffs.nunique() == 1 and diffs.iloc[0] != 0:
                issues.append("values form a perfectly uniform arithmetic sequence — "
                              "unusual for real-world measurements")

        if issues:
            findings[col] = {"issues": issues}
    return findings


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                      alpha: float = 0.05) -> dict:
    """Compare a REFERENCE dataset (e.g. last month, or the training set) to
    a CURRENT one, per shared numeric column, using the Kolmogorov-Smirnov
    two-sample test: has the distribution shifted enough to be statistically
    significant, not just random noise?

    alpha: significance threshold (0.05 = 5% false-positive rate is the
    conventional default). A lower p-value than alpha means the two samples
    likely come from different distributions -> drift detected.
    """
    shared = [c for c in reference_df.columns if c in current_df.columns
             and pd.api.types.is_numeric_dtype(reference_df[c])
             and pd.api.types.is_numeric_dtype(current_df[c])]
    findings = {}
    for col in shared:
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()
        if len(ref) < 10 or len(cur) < 10:
            continue
        statistic, p_value = stats.ks_2samp(ref, cur)
        drifted = p_value < alpha
        findings[col] = {
            "drift_detected": bool(drifted),
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "reference_mean": round(float(ref.mean()), 3),
            "current_mean": round(float(cur.mean()), 3),
            "mean_shift_pct": round(
                100 * float(cur.mean() - ref.mean()) / float(ref.mean()), 2) if ref.mean() else None,
        }
    drifted_cols = [c for c, r in findings.items() if r["drift_detected"]]
    return {
        "alpha": alpha, "columns_compared": shared,
        "columns_with_drift": drifted_cols,
        "detail": findings,
    }
