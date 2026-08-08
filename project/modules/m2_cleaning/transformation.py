"""
MODULE 2 · TRANSFORMATION ENGINE
================================
Prepares CLEANED data for downstream modelling: scaling, encoding, and simple
feature extraction. This is a distinct step from cleaning — cleaning makes data
CORRECT; transformation makes data USABLE BY A MODEL.

  1. Scaling            -> put numeric columns on a comparable range
                            (standardisation: mean 0, std 1)
  2. Encoding           -> turn categorical values into numbers
                            (one-hot for low-cardinality; ordinal for ordered categories)
  3. Feature extraction -> derive new, useful columns from existing ones
                            (e.g. a date -> day-of-week, month; a ratio between
                            two numeric columns)

WHY THIS IS SEPARATE FROM cleaning/normalisation:
Normalisation (Module 2's other file) makes a column internally CONSISTENT
(one date format, one spelling). Transformation changes a column's REPRESENTATION
for a model (a date becomes a number; a category becomes several 0/1 columns).
A dataset can be perfectly clean and still need transformation before it's fed
to ML — and conversely, transformation is USELESS on dirty data, so it always
runs after cleaning, never instead of it.

WHO NEEDS THIS: an analyst producing a Power BI dashboard or SQL report
typically does NOT need one-hot encoding — a human reads "Oxfordshire", not a
0/1 vector. This engine exists because the brief lists it, and because Module 3
(especially any ML expansion) may want scaled numeric input. It is optional in
the main pipeline for that reason (documented explicitly, not hidden).

Public entry point:  transform(df, report=None) -> (df, log)
"""
from __future__ import annotations

import pandas as pd

MAX_ONEHOT_CATEGORIES = 15   # above this, one-hot creates too many columns


def scale_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Standardise numeric columns: (x - mean) / std, so every column has mean
    0 and standard deviation 1. Puts columns with very different natural scales
    (e.g. tenure_months 1-72 vs monthly_charges 20-90) on comparable footing.

    New columns are added with a '_scaled' suffix; ORIGINAL columns are kept,
    so nothing is silently overwritten and a human-readable version survives.
    """
    df = df.copy()
    targets = columns or list(df.select_dtypes(include="number").columns)
    log = {}
    for col in targets:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        std = df[col].std()
        if std == 0 or pd.isna(std):
            log[col] = {"action": "skipped", "reason": "zero or undefined variance"}
            continue
        mean = df[col].mean()
        df[f"{col}_scaled"] = (df[col] - mean) / std
        log[col] = {"action": "standardised", "mean": round(float(mean), 4),
                    "std": round(float(std), 4), "new_column": f"{col}_scaled"}
    return df, log


def encode_categorical(df: pd.DataFrame, columns: list[str] | None = None,
                       report: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-hot encode low-cardinality categorical columns.

    Only encodes columns with <= MAX_ONEHOT_CATEGORIES distinct values — a
    high-cardinality column (e.g. 500 distinct cities) would explode into 500
    new columns, which is rarely useful and often a sign the column shouldn't
    be one-hot encoded at all (frequency or target encoding would be better,
    noted as future work). Original column is KEPT alongside the new ones.
    """
    df = df.copy()
    meta = (report or {}).get("metadata", {})
    if columns is None:
        columns = [c for c in df.columns
                  if not pd.api.types.is_numeric_dtype(df[c])
                  and meta.get(c, {}).get("semantic_type") not in
                  ("email", "phone", "postcode", "identifier", "person_name", "date")
                  and meta.get(c, {}).get("inferred_type") != "datetime"]

    log = {}
    for col in columns:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique == 0 or n_unique > MAX_ONEHOT_CATEGORIES:
            log[col] = {"action": "skipped",
                        "reason": f"{n_unique} categories (limit {MAX_ONEHOT_CATEGORIES})"}
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        log[col] = {"action": "one-hot encoded", "categories": n_unique,
                    "new_columns": list(dummies.columns)}
    return df, log


def extract_date_features(df: pd.DataFrame, date_columns: list[str] | None = None,
                          report: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Derive useful numeric/categorical features from date columns:
    year, month, day-of-week, and a simple 'is_weekend' flag.

    Dates on their own are hard for a model to use directly; these derived
    features are what typically carries the signal (e.g. churn correlating
    with month of sign-up, or day-of-week of a support ticket).
    """
    df = df.copy()
    meta = (report or {}).get("metadata", {})
    if date_columns is None:
        date_columns = [c for c in df.columns
                       if meta.get(c, {}).get("semantic_type") == "date"
                       or meta.get(c, {}).get("inferred_type") == "datetime"]
        if not date_columns:
            # no report available: fall back to sniffing values directly
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    continue
                sample = df[c].dropna().astype(str)
                if len(sample) == 0:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.8:
                    date_columns.append(c)

    log = {}
    for col in date_columns:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().sum() == 0:
            log[col] = {"action": "skipped", "reason": "no parseable dates"}
            continue
        df[f"{col}_year"] = parsed.dt.year
        df[f"{col}_month"] = parsed.dt.month
        df[f"{col}_dayofweek"] = parsed.dt.dayofweek
        df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        log[col] = {"action": "extracted date features",
                    "new_columns": [f"{col}_year", f"{col}_month",
                                    f"{col}_dayofweek", f"{col}_is_weekend"]}
    return df, log


def extract_ratio_feature(df: pd.DataFrame, numerator: str, denominator: str,
                          new_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Derive a ratio feature, e.g. monthly_charges / tenure_months = spend rate.

    Ratios often carry more business meaning than either raw column alone —
    "charges per month of tenure" is a more useful churn signal than either
    charges or tenure in isolation.
    """
    df = df.copy()
    if numerator not in df.columns or denominator not in df.columns:
        return df, {"action": "skipped", "reason": "column(s) not found"}
    name = new_name or f"{numerator}_per_{denominator}"
    ratio = df[numerator] / df[denominator].replace(0, pd.NA)
    df[name] = ratio.replace([float("inf"), float("-inf")], pd.NA)
    return df, {"action": "extracted ratio", "new_column": name,
                "formula": f"{numerator} / {denominator}"}


def transform(df: pd.DataFrame, report: dict | None = None,
             scale: bool = True, encode: bool = True,
             extract_dates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the transformation engine. All steps ADD columns; none overwrite
    the cleaned original data, so the output remains human-readable too."""
    log = {}
    if scale:
        df, log["scaling"] = scale_numeric(df)
    if encode:
        df, log["encoding"] = encode_categorical(df, report=report)
    if extract_dates:
        df, log["date_features"] = extract_date_features(df, report=report)
    return df, log
