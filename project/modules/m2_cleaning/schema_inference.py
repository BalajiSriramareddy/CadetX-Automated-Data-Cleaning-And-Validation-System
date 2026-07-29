"""
MODULE 2 · SCHEMA INFERENCE ENGINE
==================================
Infers what each column SHOULD look like — its expected type, range and format
— by learning from the data itself. The result is a schema: a machine-readable
description of the dataset's structure that later stages can check against.

Three things per column (from the brief):
  1. Expected type    -> numeric / integer / float / datetime / categorical / text / boolean
  2. Expected range   -> observed [min, max] for numeric columns
  3. Expected format  -> email / uk_phone / uk_postcode / date  (from semantic type)

Plus, usefully: allowed values for categoricals, and whether the column is
nullable (contains any missing values).

Why is this in the CLEANING module and not just reusing Module 1? Because a
schema is an *expectation* you clean and validate against. Module 1 describes
what the data IS; this turns that description into rules for what it SHOULD be.

Design principle: DATASET-AGNOSTIC. Uses Module 1's report when available, else
infers locally. Never keyed on hardcoded column names.

Public entry point:  infer_schema(df, report=None) -> dict
"""
from __future__ import annotations

import re

import pandas as pd

NUMERIC_LIKE_RE = re.compile(r"^[£$€]?\s*-?[\d,]+(\.\d+)?\s*%?$")
DATE_HINT_RE = re.compile(r"[/\-]|[A-Za-z]{3}")

SEMANTIC_TO_FORMAT = {"email": "email", "phone": "uk_phone",
                      "postcode": "uk_postcode", "date": "date"}
MAX_CATEGORY_UNIQUE = 20
MATCH_THRESHOLD = 0.8


def _local_type(series: pd.Series) -> str:
    """Light type inference used when no Module 1 report is provided."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    lowered = set(non_null.astype(str).str.strip().str.lower().unique())
    if lowered <= {"true", "false", "yes", "no", "y", "n", "0", "1"} and len(lowered) <= 3:
        return "boolean"
    as_str = non_null.astype(str).str.strip()
    if as_str.str.match(NUMERIC_LIKE_RE).mean() >= MATCH_THRESHOLD:
        # integer vs float
        cleaned = pd.to_numeric(as_str.str.replace(r"[£$€,%\s]", "", regex=True),
                                errors="coerce")
        if cleaned.dropna().apply(float.is_integer).all():
            return "integer"
        return "float"
    if as_str.str.contains(DATE_HINT_RE).mean() >= MATCH_THRESHOLD:
        parsed = pd.to_datetime(as_str, errors="coerce", format="mixed")
        if parsed.notna().mean() >= MATCH_THRESHOLD:
            return "datetime"
    if non_null.nunique() <= max(MAX_CATEGORY_UNIQUE, 0.05 * len(non_null)):
        return "categorical"
    return "text"


def _numeric_range(series: pd.Series) -> dict | None:
    """Observed [min, max] for a numeric column (after stripping symbols)."""
    cleaned = pd.to_numeric(
        series.dropna().astype(str).str.replace(r"[£$€,%\s]", "", regex=True),
        errors="coerce").dropna()
    if len(cleaned) == 0:
        return None
    return {"min": round(float(cleaned.min()), 4), "max": round(float(cleaned.max()), 4)}


def infer_schema(df: pd.DataFrame, report: dict | None = None) -> dict:
    """Infer a per-column schema: expected type, range, format, allowed values."""
    schema: dict[str, dict] = {}
    meta = (report or {}).get("metadata", {})

    for col in df.columns:
        info = meta.get(col, {})
        semantic = info.get("semantic_type")
        expected_type = info.get("inferred_type") or _local_type(df[col])

        col_schema: dict = {
            "expected_type": expected_type,
            "nullable": bool(df[col].isna().any()),
        }

        # expected range for numeric columns
        if expected_type in ("integer", "float", "numeric_as_text"):
            rng = _numeric_range(df[col])
            if rng:
                col_schema["expected_range"] = rng

        # expected format from semantic type
        if semantic in SEMANTIC_TO_FORMAT:
            col_schema["expected_format"] = SEMANTIC_TO_FORMAT[semantic]
        elif expected_type == "datetime":
            col_schema["expected_format"] = "date"

        # allowed values for genuine categoricals (not PII/identifier)
        if expected_type in ("categorical", "boolean") and \
                semantic not in ("email", "phone", "postcode", "identifier", "person_name"):
            observed = sorted({str(v) for v in df[col].dropna().unique()})
            if 0 < len(observed) <= MAX_CATEGORY_UNIQUE:
                col_schema["allowed_values"] = observed

        schema[col] = col_schema

    return {"columns": schema,
            "n_columns": len(schema),
            "source": "inferred from data" + (" + Module 1 report" if meta else "")}