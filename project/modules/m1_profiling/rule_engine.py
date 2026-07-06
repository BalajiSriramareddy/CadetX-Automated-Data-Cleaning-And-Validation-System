"""
MODULE 1 · FILE 3 — PROFILING RULE ENGINE
=========================================
Applies rules to FLAG problems, using the metadata (File 1) and profiling
(File 2) that came before it.

Three jobs (from the brief):
  1. Detect potential PII columns   -> detect_pii()
  2. Detect inconsistent formats    -> detect_inconsistent_formats()
  3. Detect suspicious columns      -> detect_suspicious_columns()

Design principle: DATASET-AGNOSTIC. Rules are driven by semantic type and
value patterns (from File 1), never by hardcoded column names. Reuses the
regex patterns and detectors from metadata.py — one source of truth.

Public entry point:  run_rules(df) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from metadata import (  # noqa: E402
    detect_semantic_type, detect_mixed_types,
    EMAIL_RE, UK_PHONE_RE, UK_POSTCODE_RE,
)

# --- thresholds ---
HIGH_MISSING_PCT = 50.0   # a column more than half empty is suspicious
HIGH_CARDINALITY = 0.9    # >90% unique -> likely an identifier / free text
MAX_EXAMPLES = 5          # how many bad examples to show per flag

# semantic types that are considered personal / sensitive
PII_TYPES = {"email", "phone", "postcode", "person_name", "identifier"}

# column-NAME hints — trigger checks even when a column is too dirty for
# value-pattern detection to fire. Biased toward recall: catch suspected
# PII/formats.
NAME_HINTS = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "mobile", "tel", "contact_number", "contact_no"],
    "postcode": ["postcode", "post_code", "postal", "zip"],
    "date": ["date", "_at", "_on", "timestamp", "dob", "birth"],
}

# which regex validates which semantic type (for format checking)
FORMAT_PATTERNS = {
    "email": EMAIL_RE,
    "phone": UK_PHONE_RE,
    "postcode": UK_POSTCODE_RE,
}


def expected_type(series: pd.Series, column_name: str) -> str:
    """Best guess of what a column SHOULD be.

    Checks the column name first (so a dirty 'email' column is still treated
    as email), then falls back to value-pattern semantic detection.
    """
    name = column_name.lower()
    for t, hints in NAME_HINTS.items():
        if any(h in name for h in hints):
            return t
    return detect_semantic_type(series, column_name)


def detect_pii(df: pd.DataFrame) -> dict:
    """Flag columns that likely hold personal or sensitive data.

    Uses semantic type (value-pattern based), so it catches an email column
    even if it were named something unhelpful.
    """
    flagged = {}
    for col in df.columns:
        sem = expected_type(df[col], col)
        if sem in PII_TYPES:
            flagged[col] = {
                "semantic_type": sem,
                "reason": f"values or name suggest {sem.replace('_', ' ')}",
            }
    return flagged


def _date_shape(value: str) -> str:
    """Classify a date-ish string into a coarse format shape."""
    v = value.strip()
    if pd.Series([v]).str.match(r"^\d{4}-\d{2}-\d{2}$").iloc[0]:
        return "YYYY-MM-DD"
    if pd.Series([v]).str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").iloc[0]:
        return "DD/MM/YYYY"
    if pd.Series([v]).str.match(r"^[A-Za-z]{3,}\s+\d{4}$").iloc[0]:
        return "Mon YYYY"
    return "other"


def detect_inconsistent_formats(df: pd.DataFrame) -> dict:
    """Find values that break the format expected for their semantic type.

    - email/phone/postcode -> values that fail the pattern
    - dates                -> presence of more than one date format shape
    """
    out = {}
    for col in df.columns:
        non_null = df[col].dropna().astype(str).str.strip()
        if len(non_null) == 0:
            continue
        sem = expected_type(df[col], col)

        # pattern-based formats (email/phone/postcode)
        if sem in FORMAT_PATTERNS:
            pattern = FORMAT_PATTERNS[sem]
            invalid_mask = ~non_null.str.match(pattern)
            n_invalid = int(invalid_mask.sum())
            if n_invalid:
                out[col] = {
                    "expected_format": sem,
                    "invalid_count": n_invalid,
                    "invalid_pct": round(100 * n_invalid / len(non_null), 2),
                    "examples": non_null[invalid_mask].unique()[:MAX_EXAMPLES].tolist(),
                }

        # date format consistency
        elif sem == "date":
            shapes = non_null.map(_date_shape)
            distinct = [s for s in shapes.unique() if s != "other"]
            if len(distinct) > 1:
                out[col] = {
                    "expected_format": "consistent date format",
                    "issue": "multiple date formats present",
                    "formats_found": distinct,
                }
    return out


def detect_suspicious_columns(df: pd.DataFrame) -> dict:
    """Flag columns that look problematic, with a human-readable reason.

    Rules: constant (no variance), mostly-missing, mixed-type, and
    high-cardinality (likely an identifier mislabelled as data).
    """
    n = len(df)
    out = {}
    for col in df.columns:
        reasons = []

        nunique = df[col].nunique(dropna=True)
        missing_pct = 100 * df[col].isna().sum() / n if n else 0
        ratio = nunique / n if n else 0

        if nunique <= 1:
            reasons.append("constant column (no variance — carries no information)")
        if missing_pct > HIGH_MISSING_PCT:
            reasons.append(f"{missing_pct:.0f}% missing (mostly empty)")
        if detect_mixed_types(df[col])["is_mixed"]:
            reasons.append("mixed value types (e.g. numbers stored alongside text)")
        if ratio > HIGH_CARDINALITY and nunique > 1:
            reasons.append("very high cardinality (likely an identifier or free text)")

        if reasons:
            out[col] = {"reasons": reasons}
    return out


def run_rules(df: pd.DataFrame) -> dict:
    """Run all three rule sets and return one dict."""
    return {
        "potential_pii": detect_pii(df),
        "inconsistent_formats": detect_inconsistent_formats(df),
        "suspicious_columns": detect_suspicious_columns(df),
    }