"""
MODULE 2 · FILE 3 — NORMALISATION
=================================
Standardises values so a column is internally consistent.

Three jobs:
  1. Coerce mixed-type numerics -> strip £ , % and convert to real numbers
                                    (fixes the 'monthly_charges = "£44.99"' problem)
  2. Standardise dates          -> parse mixed formats to one ISO format (YYYY-MM-DD)
  3. Unify categories           -> map "oxfordshire" / "OXFORDSHIRE" / " Oxfordshire "
                                    to one canonical spelling (the most frequent form)

Design principle: DATASET-AGNOSTIC. Target columns are chosen from Module 1's
report (inferred types / mixed-type flags) when available, else from light
inline heuristics. Never keyed on hardcoded column names.

Public entry point:  normalise(df, report=None) -> (df, log)
"""
from __future__ import annotations

import re
import warnings

import pandas as pd

# a value that is a number wrapped in currency / thousands / percent symbols
NUMERIC_LIKE_RE = re.compile(r"^[£$€]?\s*-?[\d,]+(\.\d+)?\s*%?$")
STRIP_CHARS_RE = re.compile(r"[£$€,%\s]")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
MATCH_THRESHOLD = 0.8


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column of possibly-mixed date formats to datetime, carefully.

    Critical: ISO dates (YYYY-MM-DD) are parsed WITHOUT dayfirst, and UK-style
    (DD/MM/YYYY, month names) WITH dayfirst. A single global dayfirst flag would
    silently swap month/day on ISO dates — a data-corruption bug this avoids.
    """
    s = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso_mask = s.str.match(ISO_DATE_RE) & series.notna()
    other_mask = ~iso_mask & series.notna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if iso_mask.any():
            result.loc[iso_mask] = pd.to_datetime(
                s[iso_mask], errors="coerce", format="mixed")
        if other_mask.any():
            result.loc[other_mask] = pd.to_datetime(
                s[other_mask], errors="coerce", dayfirst=True, format="mixed")
    return result


# ---------- target selection ----------

def _numeric_targets(df: pd.DataFrame, report: dict | None) -> list[str]:
    """Columns that look numeric but are stored as text / mixed type."""
    targets = []
    meta = (report or {}).get("metadata", {})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        if info.get("inferred_type") == "numeric_as_text" or \
           info.get("mixed_type", {}).get("is_mixed"):
            targets.append(col); continue
        # fallback heuristic when no report: mostly numeric-like strings
        sample = df[col].dropna().astype(str).str.strip()
        if len(sample) and sample.str.match(NUMERIC_LIKE_RE).mean() >= MATCH_THRESHOLD:
            targets.append(col)
    return targets


def _date_targets(df: pd.DataFrame, report: dict | None) -> list[str]:
    """Columns Module 1 considered dates (or that parse as dates)."""
    targets = []
    meta = (report or {}).get("metadata", {})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        if info.get("semantic_type") == "date" or info.get("inferred_type") == "datetime":
            targets.append(col); continue
        # fallback when no report: does the column mostly parse as dates?
        sample = df[col].dropna().astype(str)
        if len(sample) and _parse_dates(sample).notna().mean() >= MATCH_THRESHOLD:
            targets.append(col)
    return targets


def _category_targets(df: pd.DataFrame, report: dict | None) -> list[str]:
    """Low-cardinality text columns — safe to canonicalise spelling/case.
    Excludes identifiers, emails, and person names (case matters there)."""
    targets = []
    meta = (report or {}).get("metadata", {})
    n = len(df)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        sem = info.get("semantic_type")
        if sem in ("email", "identifier", "person_name", "phone"):
            continue
        inferred = info.get("inferred_type")
        is_categorical = inferred == "categorical" or \
            (inferred is None and df[col].nunique(dropna=True) <= max(20, 0.05 * n))
        if is_categorical:
            targets.append(col)
    return targets


# ---------- transforms ----------

def coerce_numeric(df: pd.DataFrame, report: dict | None, log: list) -> pd.DataFrame:
    for col in _numeric_targets(df, report):
        cleaned = df[col].astype(str).str.replace(STRIP_CHARS_RE, "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        failed = int(converted.isna().sum() - df[col].isna().sum())
        df[col] = converted
        log.append({"column": col, "action": "coerced to numeric (stripped £,%)",
                    "failed_conversions": max(failed, 0)})
    return df


def standardise_dates(df: pd.DataFrame, report: dict | None, log: list) -> pd.DataFrame:
    for col in _date_targets(df, report):
        parsed = _parse_dates(df[col])
        failed = int(parsed.isna().sum() - df[col].isna().sum())
        df[col] = parsed.dt.strftime("%Y-%m-%d")
        log.append({"column": col, "action": "standardised dates to YYYY-MM-DD",
                    "unparseable": max(failed, 0)})
    return df


def _canonical_map(series: pd.Series) -> dict:
    """Map each case/space-insensitive key to its best original form.
    Prefers the most frequent spelling; on ties, prefers a properly-cased
    form (not ALL CAPS, not all lower), else a Title-Cased version."""
    s = series.dropna().astype(str).str.strip()
    tmp = pd.DataFrame({"orig": s, "key": s.str.lower()})
    mapping = {}
    for k, g in tmp.groupby("key"):
        counts = g["orig"].value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            mapping[k] = top[0]
        else:
            nice = [t for t in top if not t.isupper() and not t.islower()]
            mapping[k] = nice[0] if nice else k.title()
    return mapping


def unify_categories(df: pd.DataFrame, report: dict | None, log: list) -> pd.DataFrame:
    for col in _category_targets(df, report):
        before = df[col].nunique(dropna=True)
        mapping = _canonical_map(df[col])
        df[col] = df[col].map(
            lambda v: mapping.get(str(v).strip().lower(), v) if pd.notna(v) else v
        )
        after = df[col].nunique(dropna=True)
        if after < before:
            log.append({"column": col,
                        "action": "unified category spelling/case",
                        "distinct_before": before, "distinct_after": after})
    return df


def normalise(df: pd.DataFrame, report: dict | None = None):
    """Run all three normalisation steps. Returns (df, log)."""
    df = df.copy()
    log = []
    df = coerce_numeric(df, report, log)
    df = standardise_dates(df, report, log)
    df = unify_categories(df, report, log)
    return df, log