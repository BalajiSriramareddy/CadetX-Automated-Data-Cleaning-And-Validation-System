"""
MODULE 2 · FILE 2 — DEDUPLICATION
=================================
Finds and removes duplicate rows, and flags key conflicts.

Three levels (increasingly strict about what counts as a duplicate):
  1. EXACT duplicates        -> identical rows                 (remove)
  2. NORMALISED duplicates   -> identical after trimming/lower  (remove)
     e.g. "John Smith" and "  john smith " are the same person
  3. KEY conflicts           -> same identifier, different data (FLAG, don't remove)
     e.g. two rows share customer_id but disagree — a human must decide

Why flag key conflicts instead of deleting? Deleting one could discard correct
data. A gatekeeper surfaces the conflict for review rather than guessing.

True fuzzy matching (typo-level, e.g. "Jon" vs "John") is noted as future work —
it risks false merges, so it needs care beyond the core build.

Design principle: DATASET-AGNOSTIC. Works on whatever columns exist.

Public entry point:  remove_duplicates(df, report=None) -> (df, log)
"""
from __future__ import annotations

import pandas as pd


def _normalised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy with text columns trimmed and lower-cased — for COMPARISON only.
    The original values are what get kept; this is just how we spot matches.

    Note: we test 'is this NOT numeric/datetime' rather than 'is dtype object',
    because newer pandas reports text columns as dtype 'str', not 'object' — a
    version gotcha that silently skips normalisation if you check == object.
    """
    norm = df.copy()
    for col in norm.columns:
        if not (pd.api.types.is_numeric_dtype(norm[col])
                or pd.api.types.is_datetime64_any_dtype(norm[col])):
            norm[col] = norm[col].astype(str).str.strip().str.lower()
    return norm


def _identifier_columns(df: pd.DataFrame, report: dict | None) -> list[str]:
    """Columns Module 1 tagged as identifiers (else empty)."""
    if not report or "metadata" not in report:
        return []
    return [c for c, m in report["metadata"].items()
            if m.get("semantic_type") == "identifier" and c in df.columns]


def remove_duplicates(df: pd.DataFrame, report: dict | None = None):
    """Remove exact + normalised duplicates; flag key conflicts. Returns (df, log)."""
    df = df.copy()
    start_rows = len(df)
    log = []

    # 1. exact duplicates
    exact_mask = df.duplicated(keep="first")
    n_exact = int(exact_mask.sum())
    if n_exact:
        df = df[~exact_mask]
        log.append({"step": "exact_duplicates", "removed": n_exact})

    # 2. normalised duplicates (formatting-only differences)
    norm = _normalised_frame(df)
    norm_mask = norm.duplicated(keep="first")
    n_norm = int(norm_mask.sum())
    if n_norm:
        df = df[~norm_mask.values]
        log.append({"step": "normalised_duplicates", "removed": n_norm})

    # 3. key conflicts — same identifier value appearing more than once (flag only)
    key_conflicts = {}
    for col in _identifier_columns(df, report):
        dupe_ids = df[col][df[col].duplicated(keep=False)].dropna().unique()
        if len(dupe_ids):
            key_conflicts[col] = {
                "conflicting_values": len(dupe_ids),
                "examples": [str(v) for v in dupe_ids[:5]],
            }
    if key_conflicts:
        log.append({"step": "key_conflicts_flagged", "detail": key_conflicts})

    log.append({
        "step": "summary",
        "rows_before": start_rows,
        "rows_after": len(df),
        "rows_removed": start_rows - len(df),
    })
    return df, log