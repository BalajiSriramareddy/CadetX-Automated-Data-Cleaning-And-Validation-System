"""
MODULE 2 · FILE 3 — NORMALISATION
=================================
Standardises values so a column is internally consistent.

Jobs:
  1. Coerce mixed-type numerics -> strip currency/% and convert to real numbers
     ...BUT never coerce identifier-like columns (phone/postcode), because that
     destroys leading zeros (07123... -> 7123...). Identifier safety guard below.
  2. Standardise dates          -> parse mixed formats to one ISO format (YYYY-MM-DD),
     parsing ISO and UK-style dates separately so ISO dates aren't corrupted.
  3. Unify binary categories    -> Yes/Y/1/true -> "Yes",  No/N/0/false -> "No"
  4. Unify general categories   -> map spelling/case variants to one canonical form

Design principle: DATASET-AGNOSTIC. Targets come from Module 1's report
(inferred/semantic types) when available, else light inline heuristics.

Public entry point:  normalise(df, report=None) -> (df, log)
"""
from __future__ import annotations

import re
import warnings

import pandas as pd

NUMERIC_LIKE_RE = re.compile(r"^[£$€]?\s*-?[\d,]+(\.\d+)?\s*%?$")
STRIP_CHARS_RE = re.compile(r"[£$€,%\s]")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
LEADING_ZERO_RE = re.compile(r"^0\d+$")   # 07123..., identifier not a number
BARE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")  # 892, 0, 44.99 -> a number, NOT a date
# a real date has separators or a month name; bare integers must not be parsed as
# years (pandas reads "1000" as year 1000, which overflows its datetime range).
DATE_HINT_RE = re.compile(r"[/\-]|[A-Za-z]{3}")
MATCH_THRESHOLD = 0.8

# semantic types that must NEVER be coerced to numbers even if they look numeric
IDENTIFIER_SEMANTICS = {"phone", "postcode", "identifier", "person_name", "email"}

_YES_SET = {"yes", "y", "1", "true", "t"}
_NO_SET = {"no", "n", "0", "false", "f"}


def _looks_like_dates(series: pd.Series) -> bool:
    """A column is a date candidate only if its values look date-ish: they
    contain a separator or month name. Bare numbers (892, 0, 1) are NOT dates,
    even though pandas would parse '1000' as the year 1000 (and overflow)."""
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return False
    if s.str.match(BARE_NUMBER_RE).mean() >= 0.5:   # mostly bare numbers -> not dates
        return False
    return s.str.contains(DATE_HINT_RE).mean() >= MATCH_THRESHOLD


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse possibly-mixed date formats, carefully and crash-proof.

    ISO parsed WITHOUT dayfirst, UK-style WITH dayfirst, so ISO dates are not
    corrupted. Wrapped in try/except so an out-of-range value (pandas can only
    represent ~1677-2262) degrades to NaT instead of crashing the pipeline.
    """
    s = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso_mask = s.str.match(ISO_DATE_RE) & series.notna()
    other_mask = ~iso_mask & series.notna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if iso_mask.any():
                result.loc[iso_mask] = pd.to_datetime(
                    s[iso_mask], errors="coerce", format="mixed")
            if other_mask.any():
                result.loc[other_mask] = pd.to_datetime(
                    s[other_mask], errors="coerce", dayfirst=True, format="mixed")
        except Exception:
            # any parsing failure (e.g. out-of-bounds date) -> treat as unparseable
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return result


def _looks_like_identifier(series: pd.Series) -> bool:
    """No-report guard: leading-zero numbers (phones) are identifiers, not numbers."""
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return False
    return s.str.match(LEADING_ZERO_RE).mean() >= 0.3


def _numeric_targets(df, report):
    targets = []
    meta = (report or {}).get("metadata", {})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        if info.get("semantic_type") in IDENTIFIER_SEMANTICS:   # guard 1: report
            continue
        if _looks_like_identifier(df[col]):                     # guard 2: no report
            continue
        # A column is only coerced to numeric if its values are PREDOMINANTLY
        # numeric-like. The Module 1 "mixed_type" flag alone is NOT enough:
        # a mostly-text column containing one number-like value (e.g. a postcode
        # column holding "12345" among real postcodes) is flagged mixed, and
        # coercing it would turn every genuine text value into NaN.
        sample = df[col].dropna().astype(str).str.strip()
        if len(sample) == 0:
            continue
        numeric_ratio = sample.str.match(NUMERIC_LIKE_RE).mean()
        if numeric_ratio < MATCH_THRESHOLD:
            continue
        targets.append(col)
    return targets


def _date_targets(df, report):
    targets = []
    meta = (report or {}).get("metadata", {})
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        if info.get("semantic_type") == "date" or info.get("inferred_type") == "datetime":
            targets.append(col); continue
        # only test parsing on columns that actually LOOK like dates (have
        # separators / month names). This stops bare-integer columns such as
        # PassengerId or a 0/1 flag from being misread as years.
        if not _looks_like_dates(df[col]):
            continue
        sample = df[col].dropna().astype(str)
        if len(sample) and _parse_dates(sample).notna().mean() >= MATCH_THRESHOLD:
            targets.append(col)
    return targets


def _binary_targets(df):
    targets = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        distinct = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if distinct and distinct.issubset(_YES_SET | _NO_SET):
            targets.append(col)
    return targets


def _category_targets(df, report, exclude):
    targets = []
    meta = (report or {}).get("metadata", {})
    n = len(df)
    for col in df.columns:
        if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
            continue
        info = meta.get(col, {})
        if info.get("semantic_type") in ("email", "identifier", "person_name", "phone"):
            continue
        inferred = info.get("inferred_type")
        is_categorical = inferred == "categorical" or \
            (inferred is None and df[col].nunique(dropna=True) <= max(20, 0.05 * n))
        if is_categorical:
            targets.append(col)
    return targets


def coerce_numeric(df, report, log):
    for col in _numeric_targets(df, report):
        cleaned = df[col].astype(str).str.replace(STRIP_CHARS_RE, "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        failed = int(converted.isna().sum() - df[col].isna().sum())
        df[col] = converted
        log.append({"column": col, "action": "coerced to numeric (stripped currency/%)",
                    "failed_conversions": max(failed, 0)})
    return df


def standardise_dates(df, report, log):
    for col in _date_targets(df, report):
        parsed = _parse_dates(df[col])
        failed = int(parsed.isna().sum() - df[col].isna().sum())
        df[col] = parsed.dt.strftime("%Y-%m-%d")
        log.append({"column": col, "action": "standardised dates to YYYY-MM-DD",
                    "unparseable": max(failed, 0)})
    return df


def unify_binary(df, log):
    def _map(v):
        if pd.isna(v):
            return v
        low = str(v).strip().lower()
        if low in _YES_SET:
            return "Yes"
        if low in _NO_SET:
            return "No"
        return v
    binary_cols = _binary_targets(df)
    for col in binary_cols:
        before = sorted({str(v) for v in df[col].dropna().unique()})[:6]
        df[col] = df[col].map(_map)
        log.append({"column": col, "action": "unified Yes/No-style values",
                    "distinct_before": before})
    return df, set(binary_cols)


def _canonical_map(series):
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


def unify_categories(df, report, log, exclude):
    for col in _category_targets(df, report, exclude):
        before = df[col].nunique(dropna=True)
        mapping = _canonical_map(df[col])
        df[col] = df[col].map(
            lambda v: mapping.get(str(v).strip().lower(), v) if pd.notna(v) else v)
        after = df[col].nunique(dropna=True)
        if after < before:
            log.append({"column": col, "action": "unified category spelling/case",
                        "distinct_before": before, "distinct_after": after})
    return df


def normalise(df, report=None):
    """Run all normalisation steps. Returns (df, log)."""
    df = df.copy()
    log = []
    df = coerce_numeric(df, report, log)
    df = standardise_dates(df, report, log)
    df, binary_cols = unify_binary(df, log)
    df = unify_categories(df, report, log, exclude=binary_cols)
    return df, log