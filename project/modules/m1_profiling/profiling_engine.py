"""
MODULE 1 · FILE 2 — PROFILING ENGINE
====================================
Profiles a dataset in depth, AFTER the metadata engine has labelled it.

Four jobs (from the brief):
  1. Missing-value matrix        -> missing_value_analysis()
  2. Data-type consistency check -> type_consistency()
  3. Unique value & cardinality  -> cardinality_analysis()
  4. Correlation & distributions -> correlation_matrix(), distributions()

Design principle: DATASET-AGNOSTIC. Everything iterates over df.columns and
branches on dtype / inferred type, never on hardcoded column names. Reuses
infer_type() from the metadata engine so type logic lives in one place.

Public entry point:  profile_data(df) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# import infer_type from File 1 (works standalone and inside the package)
sys.path.append(str(Path(__file__).resolve().parent))
from metadata import infer_type  # noqa: E402

TOP_N = 5  # how many top values to report for categorical columns


def missing_value_analysis(df: pd.DataFrame) -> dict:
    """Per-column and overall missing-value summary (the 'missing-value matrix').

    The visual heatmap is built in File 4 from these same numbers.
    """
    per_column = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        per_column[col] = {
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / len(df), 2),
        }
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    return {
        "per_column": per_column,
        "overall": {
            "total_missing_cells": total_missing,
            "total_cells": int(total_cells),
            "missing_pct": round(100 * total_missing / total_cells, 2),
            "complete_columns": [c for c in df.columns if df[c].notna().all()],
            "empty_columns": [c for c in df.columns if df[c].isna().all()],
        },
    }


def type_consistency(df: pd.DataFrame) -> dict:
    """For each column, what fraction of values match the column's inferred type.

    A low score flags a column whose values disagree with each other
    (e.g. numbers stored alongside '£44.99' strings).
    """
    out = {}
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            out[col] = {"inferred_type": "empty", "consistency_pct": None,
                        "nonconforming": 0}
            continue

        inferred = infer_type(df[col])
        # count values that can be coerced to the inferred type family
        if inferred in ("integer", "float", "numeric_as_text"):
            coerced = pd.to_numeric(
                non_null.astype(str).str.replace(r"[£$€,%\s]", "", regex=True),
                errors="coerce",
            )
            conforming = coerced.notna().sum()
        elif inferred == "datetime":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                coerced = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
            conforming = coerced.notna().sum()
        else:
            # categorical/boolean/text: every non-null value "conforms"
            conforming = len(non_null)

        nonconforming = int(len(non_null) - conforming)
        out[col] = {
            "inferred_type": inferred,
            "consistency_pct": round(100 * conforming / len(non_null), 2),
            "nonconforming": nonconforming,
        }
    return out


def cardinality_analysis(df: pd.DataFrame) -> dict:
    """Unique-value counts, cardinality ratio, and flags for constant / id-like."""
    out = {}
    n = len(df)
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        ratio = round(nunique / n, 4) if n else 0.0
        out[col] = {
            "unique_values": nunique,
            "cardinality_ratio": ratio,
            "is_constant": nunique <= 1,            # same value everywhere -> useless
            "is_high_cardinality": ratio > 0.9,     # likely an identifier / free text
        }
    return out


def correlation_matrix(df: pd.DataFrame) -> dict:
    """Pearson correlation between numeric columns.

    Note: mixed-type columns stored as text (e.g. '£44.99') are NOT included
    here — they are excluded until the cleaning module converts them. That is
    intentional: profiling reports on the data as it actually is.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"note": "fewer than 2 numeric columns; no correlation computed"}
    corr = numeric.corr(numeric_only=True).round(3)
    return corr.where(pd.notnull(corr), None).to_dict()


def distributions(df: pd.DataFrame) -> dict:
    """Distribution summary per column.

    Numeric -> min/max/mean/median/std/quartiles/skew.
    Categorical/text -> the TOP_N most frequent values and their counts.
    """
    out = {}
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            out[col] = {"kind": "empty"}
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = {
                "kind": "numeric",
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": round(float(non_null.mean()), 3),
                "median": float(non_null.median()),
                "std": round(float(non_null.std()), 3),
                "q1": float(non_null.quantile(0.25)),
                "q3": float(non_null.quantile(0.75)),
                "skew": round(float(non_null.skew()), 3) if len(non_null) > 2 else None,
            }
        else:
            counts = non_null.astype(str).value_counts().head(TOP_N)
            out[col] = {
                "kind": "categorical",
                "top_values": {str(k): int(v) for k, v in counts.items()},
            }
    return out


def profile_data(df: pd.DataFrame) -> dict:
    """Run the full profiling engine and return one dict."""
    return {
        "missing_values": missing_value_analysis(df),
        "type_consistency": type_consistency(df),
        "cardinality": cardinality_analysis(df),
        "correlations": correlation_matrix(df),
        "distributions": distributions(df),
    }