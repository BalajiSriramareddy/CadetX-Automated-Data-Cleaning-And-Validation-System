"""
MODULE 1 · FILE 1 — METADATA EXTRACTION ENGINE
==============================================
Understands WHAT each column is, before any profiling or cleaning.

Three jobs (from the brief):
  1. Infer column types        -> infer_type()
  2. Detect semantic meaning   -> detect_semantic_type()   (name, email, date, id...)
  3. Detect mixed-type columns -> detect_mixed_types()

Design principle: DATASET-AGNOSTIC. Decisions are driven by the column's
*values* (patterns, dtypes), not by hardcoded column names. Column names are
only used as a soft hint. This is what lets the system run on any dataset
with zero changes.

Public entry point:  extract_metadata(df) -> dict   (one record per column)
"""
from __future__ import annotations

import re
import pandas as pd

# --- value patterns used for semantic detection (not name-based) ---
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
UK_PHONE_RE = re.compile(r"^(0|\+44)\d{9,10}$")
UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$", re.IGNORECASE)
# a number optionally wrapped in currency symbols / thousands separators
NUMERIC_LIKE_RE = re.compile(r"^[£$€]?\s*-?[\d,]+(\.\d+)?\s*%?$")

# how big a sample we test patterns against (keeps it fast on large data)
SAMPLE_SIZE = 500
# fraction of values that must match a pattern to accept it
MATCH_THRESHOLD = 0.8


def _clean_sample(series: pd.Series) -> pd.Series:
    """Drop nulls, cast to string, take a sample for pattern testing."""
    s = series.dropna().astype(str).str.strip()
    if len(s) > SAMPLE_SIZE:
        s = s.sample(SAMPLE_SIZE, random_state=42)
    return s


def _match_ratio(sample: pd.Series, pattern: re.Pattern) -> float:
    """Fraction of sample values that match a regex."""
    if len(sample) == 0:
        return 0.0
    return sample.str.match(pattern).mean()


def infer_type(series: pd.Series) -> str:
    """Infer a friendly type label from the column's values.

    Returns one of:
      empty | boolean | integer | float | numeric_as_text |
      datetime | categorical | text
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"

    # boolean: small set of yes/no/true/false/0/1
    lowered = set(non_null.astype(str).str.strip().str.lower().unique())
    if lowered <= {"true", "false", "yes", "no", "y", "n", "0", "1"} and len(lowered) <= 3:
        return "boolean"

    # native numeric dtypes
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"

    # native datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # object column that is *actually* numeric (e.g. "£29.99")
    sample = _clean_sample(series)
    if _match_ratio(sample, NUMERIC_LIKE_RE) >= MATCH_THRESHOLD:
        return "numeric_as_text"

    # object column that parses as dates (suppress noisy format-inference warning)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce", dayfirst=True)
    if parsed.notna().mean() >= MATCH_THRESHOLD:
        return "datetime"

    # low cardinality relative to size -> categorical
    if non_null.nunique() <= max(20, 0.05 * len(non_null)):
        return "categorical"

    return "text"


def detect_semantic_type(series: pd.Series, column_name: str = "") -> str:
    """Detect what the column *means* from its values (with a soft name hint).

    Returns: email | phone | postcode | person_name | identifier |
             date | numeric | categorical | text | unknown
    """
    sample = _clean_sample(series)
    if len(sample) == 0:
        return "unknown"

    name = column_name.lower()

    # value-pattern checks first (these are reliable regardless of column name)
    if _match_ratio(sample, EMAIL_RE) >= MATCH_THRESHOLD:
        return "email"
    if _match_ratio(sample, UK_PHONE_RE) >= MATCH_THRESHOLD:
        return "phone"
    if _match_ratio(sample, UK_POSTCODE_RE) >= MATCH_THRESHOLD:
        return "postcode"

    # identifier: name hints at id AND values are highly unique
    uniqueness = series.nunique() / max(len(series.dropna()), 1)
    if ("id" in name or name.endswith("_no") or name == "code") and uniqueness > 0.9:
        return "identifier"

    # person name: name hint + two-token alphabetic values
    looks_like_name = sample.str.match(r"^[A-Za-z]+(\s[A-Za-z]+)+$").mean() >= MATCH_THRESHOLD
    if ("name" in name or looks_like_name) and "user" not in name and "file" not in name:
        if looks_like_name:
            return "person_name"

    # fall back to the inferred type family
    inferred = infer_type(series)
    if inferred in ("datetime",):
        return "date"
    if inferred in ("integer", "float", "numeric_as_text"):
        return "numeric"
    if inferred in ("categorical", "boolean"):
        return "categorical"
    return "text"


def detect_mixed_types(series: pd.Series) -> dict:
    """Flag columns whose non-null values are of inconsistent Python types.

    Catches e.g. a numeric column where some rows are "£29.99" strings.
    Returns {is_mixed, python_types, examples}.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"is_mixed": False, "python_types": [], "examples": []}

    # classify each value as number vs string-that-isn't-a-clean-number
    def base_kind(v):
        if isinstance(v, (int, float)):
            return "number"
        s = str(v).strip()
        # a string that is a clean number is still "number-like"
        return "number" if NUMERIC_LIKE_RE.match(s) and not re.search(r"[£$€%,]", s) else "string"

    kinds = non_null.map(base_kind)
    unique_kinds = sorted(kinds.unique())
    is_mixed = len(unique_kinds) > 1

    examples = []
    if is_mixed:
        # one example of each kind for the report
        for k in unique_kinds:
            examples.append(str(non_null[kinds == k].iloc[0]))

    return {
        "is_mixed": bool(is_mixed),
        "python_types": unique_kinds,
        "examples": examples,
    }


def extract_metadata(df: pd.DataFrame) -> dict:
    """Build per-column metadata for the whole dataframe.

    Returns: { column_name: {pandas_dtype, inferred_type, semantic_type,
                             mixed_type: {...}} }
    """
    meta = {}
    for col in df.columns:
        meta[col] = {
            "pandas_dtype": str(df[col].dtype),
            "inferred_type": infer_type(df[col]),
            "semantic_type": detect_semantic_type(df[col], col),
            "mixed_type": detect_mixed_types(df[col]),
        }
    return meta