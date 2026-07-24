"""
MODULE 3 · FILE 1 — RULE-BASED VALIDATION
=========================================
Checks a (cleaned) dataset against a set of RULES and reports violations.

Supported rule types (per column):
  format  : "email" | "uk_phone" | "uk_postcode"   -> value must match the pattern
  range   : {"min": 0, "max": 10}                  -> numeric value must sit in range
  allowed : ["Yes", "No"]                           -> value must be one of a set
  required: true                                     -> value must not be missing

Rules come from a spec dict, e.g.:
  {
    "email":     {"format": "email"},
    "nps_score": {"range": {"min": 0, "max": 10}},
    "churn":     {"allowed": ["Yes", "No"]},
  }

Where does the spec come from? Two layers (see config.py):
  - generic defaults auto-built from Module 1's report (works with zero config)
  - optional rules.yaml for domain rules (e.g. NPS 0-10)

This file just APPLIES a spec — it doesn't care where the spec came from, which
keeps it dataset-agnostic.

Public entry point:  validate_rules(df, rules) -> dict
"""
from __future__ import annotations

import re

import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
UK_PHONE_RE = re.compile(r"^(0|\+44)\d{9,10}$")
UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$", re.IGNORECASE)

FORMAT_PATTERNS = {
    "email": EMAIL_RE,
    "uk_phone": UK_PHONE_RE,
    "uk_postcode": UK_POSTCODE_RE,
}
MAX_EXAMPLES = 5


def _format_violations(series: pd.Series, fmt: str) -> pd.Series:
    """Boolean mask of non-null values that FAIL the format pattern."""
    pattern = FORMAT_PATTERNS.get(fmt)
    if pattern is None:
        return pd.Series(False, index=series.index)
    non_null = series.notna()
    matches = series.astype(str).str.strip().str.match(pattern)
    return non_null & ~matches


def _range_violations(series: pd.Series, lo=None, hi=None) -> pd.Series:
    """Mask of numeric values outside [lo, hi]. Non-numeric values count as invalid."""
    numeric = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & numeric.isna()          # can't even be read as a number
    if lo is not None:
        bad = bad | (numeric < lo)
    if hi is not None:
        bad = bad | (numeric > hi)
    return bad.fillna(False)


def _allowed_violations(series: pd.Series, allowed: list) -> pd.Series:
    """Mask of values not in the allowed set."""
    allowed_set = {str(a) for a in allowed}
    return series.notna() & ~series.astype(str).isin(allowed_set)


def _record(series: pd.Series, mask: pd.Series, rule: str) -> dict:
    n = int(mask.sum())
    return {
        "rule": rule,
        "violations": n,
        "violation_pct": round(100 * n / max(series.notna().sum(), 1), 2),
        "examples": series[mask].astype(str).unique()[:MAX_EXAMPLES].tolist(),
    }


def validate_rules(df: pd.DataFrame, rules: dict) -> dict:
    """Apply a rules spec to the dataframe and return per-column violations."""
    results = {}
    total = 0

    for col, spec in rules.items():
        if col not in df.columns:
            continue
        checks = []
        series = df[col]

        if "required" in spec and spec["required"]:
            mask = series.isna()
            if mask.any():
                checks.append({"rule": "required", "violations": int(mask.sum()),
                               "violation_pct": round(100 * mask.mean(), 2),
                               "examples": []})

        if "format" in spec:
            mask = _format_violations(series, spec["format"])
            if mask.any():
                checks.append(_record(series, mask, f"format:{spec['format']}"))

        if "range" in spec:
            r = spec["range"]
            mask = _range_violations(series, r.get("min"), r.get("max"))
            if mask.any():
                checks.append(_record(series, mask,
                                       f"range:[{r.get('min')},{r.get('max')}]"))

        if "allowed" in spec:
            mask = _allowed_violations(series, spec["allowed"])
            if mask.any():
                checks.append(_record(series, mask, "allowed_values"))

        if checks:
            results[col] = checks
            total += sum(c["violations"] for c in checks)

    return {
        "column_violations": results,
        "summary": {
            "total_violations": total,
            "columns_with_violations": list(results.keys()),
            "columns_checked": [c for c in rules if c in df.columns],
        },
    }