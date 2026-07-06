"""
MODULE 2 · FILE 1 — IMPUTATION ENGINE
=====================================
Fills missing values, using the profiling report from Module 1 to decide how.

Strategy (statistical, explainable — ML imputers are noted as future work):
  - numeric columns      -> fill with the MEDIAN (robust to outliers)
  - categorical/text     -> fill with the MODE (most frequent value)
  - datetime columns     -> left as-is (imputing dates needs domain logic)

Every fill is recorded in an audit log so the cleaning is fully traceable.

Design principle: DATASET-AGNOSTIC. Decisions come from each column's inferred
type (from Module 1) or its dtype, never from hardcoded column names.

Public entry point:  impute_missing(df, report=None) -> (df, log)
"""
from __future__ import annotations

import pandas as pd


def _numeric_strategy(series: pd.Series):
    """Median — robust to outliers, unlike the mean."""
    return series.median()


def _categorical_strategy(series: pd.Series):
    """Mode — the most frequent value. Returns None if the column is all-null."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def impute_missing(df: pd.DataFrame, report: dict | None = None):
    """Fill missing values by type and return (cleaned_df, audit_log).

    If a Module 1 profiling `report` is passed, its inferred types guide the
    choice; otherwise we fall back to pandas dtype. Both paths are agnostic.
    """
    df = df.copy()
    log = []

    # inferred types from Module 1,
    inferred = {}
    if report and "metadata" in report:
        inferred = {c: m.get("inferred_type") for c, m in report["metadata"].items()}

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        col_type = inferred.get(col)
        # decide numeric vs categorical, from inferred type first, then dtype
        is_numeric = (
            col_type in ("integer", "float")
            or (col_type is None and pd.api.types.is_numeric_dtype(df[col]))
        )
        is_datetime = (
            col_type == "datetime"
            or pd.api.types.is_datetime64_any_dtype(df[col])
        )

        if is_datetime:
            log.append({"column": col, "missing": n_missing,
                        "action": "skipped (datetime — needs domain logic)"})
            continue

        if is_numeric:
            fill_value = _numeric_strategy(df[col])
            strategy = "median"
        else:
            fill_value = _categorical_strategy(df[col])
            strategy = "mode"

        if fill_value is None:
            log.append({"column": col, "missing": n_missing,
                        "action": "skipped (no value available to impute)"})
            continue

        df[col] = df[col].fillna(fill_value)
        # convert numpy scalar types to native Python so the log is JSON-safe
        safe_value = fill_value.item() if hasattr(fill_value, "item") else fill_value
        log.append({
            "column": col,
            "missing": n_missing,
            "action": f"filled with {strategy}",
            "fill_value": safe_value,
        })

    return df, log