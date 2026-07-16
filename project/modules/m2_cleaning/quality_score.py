"""
MODULE 2 · FILE 4 — DATA-QUALITY SCORING
========================================
Produces a single 0-100 data-quality score, and the BEFORE vs AFTER delta that
proves the cleaning worked. This is the headline number for the whole module.

The score is a weighted average of three measurable dimensions, each of which a
specific cleaning step improves:

  completeness   (40%)  = non-missing cells        -> improved by imputation
  uniqueness     (30%)  = non-duplicate rows       -> improved by deduplication
  type_validity  (30%)  = columns with a clean type -> improved by normalisation

Why a weighted composite and not one number? Because "quality" isn't one thing.
Splitting it into dimensions makes the score explainable — you can say WHICH
aspect improved, not just that a number went up.

Design principle: DATASET-AGNOSTIC. Every dimension is computed from the data
itself, no hardcoded column names.

Public entry points:
  compute_quality_score(df) -> dict
  score_before_after(before_df, after_df) -> dict
"""
from __future__ import annotations

import re

import pandas as pd

NUMERIC_LIKE_RE = re.compile(r"^[£$€]?\s*-?[\d,]+(\.\d+)?\s*%?$")

WEIGHTS = {"completeness": 0.4, "uniqueness": 0.3, "type_validity": 0.3}


def _completeness(df: pd.DataFrame) -> float:
    """Fraction of cells that are not missing."""
    if df.size == 0:
        return 0.0
    return 1 - df.isna().sum().sum() / df.size


def _uniqueness(df: pd.DataFrame) -> float:
    """Fraction of rows that are not duplicates."""
    if len(df) == 0:
        return 0.0
    return 1 - df.duplicated().sum() / len(df)


def _column_type_clean(s: pd.Series) -> bool:
    """True if a column has a clean, consistent type.

    A real numeric/datetime/bool column is clean. An object column is clean
    only if its values are genuinely textual — NOT numbers hiding as text
    (e.g. '£44.99'), which normalisation is meant to fix.
    """
    if (pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_bool_dtype(s)):
        return True
    non_null = s.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False
    numeric_like = non_null.str.match(NUMERIC_LIKE_RE).mean()
    return numeric_like < 0.2  # mostly text = clean; mostly numeric-as-text = not


def _type_validity(df: pd.DataFrame) -> float:
    """Fraction of columns that have a clean, consistent type."""
    if df.shape[1] == 0:
        return 0.0
    clean = sum(_column_type_clean(df[c]) for c in df.columns)
    return clean / df.shape[1]


def compute_quality_score(df: pd.DataFrame) -> dict:
    """Return the overall 0-100 score plus its three components."""
    components = {
        "completeness": round(_completeness(df), 4),
        "uniqueness": round(_uniqueness(df), 4),
        "type_validity": round(_type_validity(df), 4),
    }
    overall = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return {
        "overall_score": round(100 * float(overall), 2),
        "components": {k: round(100 * float(v), 2) for k, v in components.items()},
        "weights": WEIGHTS,
    }


def score_before_after(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict:
    """Score both datasets and report the improvement delta."""
    before = compute_quality_score(before_df)
    after = compute_quality_score(after_df)
    return {
        "score_before": before["overall_score"],
        "score_after": after["overall_score"],
        "delta": round(after["overall_score"] - before["overall_score"], 2),
        "components_before": before["components"],
        "components_after": after["components"],
    }