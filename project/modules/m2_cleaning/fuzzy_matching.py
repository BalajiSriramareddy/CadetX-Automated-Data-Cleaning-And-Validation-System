"""
MODULE 2 · FUZZY DUPLICATE MATCHING
===================================
Finds NEAR-duplicate rows that exact and normalised matching miss — genuine
typos or minor variations, e.g. "Jon Smith" vs "John Smith", or a phone number
one digit off.

WHY THIS IS SEPARATE FROM deduplication.py's exact/normalised matching:
Fuzzy matching is fundamentally riskier. Exact matching can never be wrong —
identical values ARE the same record. Fuzzy matching is a JUDGEMENT CALL: two
similar-but-different values might be the same entity with a typo, or they
might be two different people who happen to have similar names. Getting this
wrong MERGES two real customers into one — a worse error than missing a
duplicate. So this function:
  - only FLAGS candidate fuzzy duplicates, it never merges or deletes rows
    automatically (unlike the exact/normalised dedup, which is safe to auto-fix)
  - only runs on genuinely name-like or free-text columns, not identifiers,
    emails, or numeric data, where "close" values are usually NOT duplicates
  - uses a high similarity threshold (default 0.88) to keep false positives low
  - always requires human review before any row is actually removed

Method: difflib.SequenceMatcher (Python standard library — no extra dependency
to install or justify). It is slower than specialised libraries on very large
datasets, which is an accepted trade-off documented below.

Public entry point:  find_fuzzy_duplicates(df, columns=None, threshold=0.88) -> dict
"""
from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

DEFAULT_THRESHOLD = 0.88   # high bar: minimise false positives (see module docstring)
MAX_COMPARISONS_PER_COLUMN = 5000   # O(n^2) safety cap; see performance note below
MAX_PAIRS_REPORTED = 20


def _similarity(a: str, b: str) -> float:
    """0.0-1.0 similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_candidate_columns(df: pd.DataFrame, report: dict | None) -> list[str]:
    """Only free-text / name-like columns are safe fuzzy-match candidates.
    Identifiers, emails, phone numbers, and numeric data are excluded: a
    'close' value there is usually a DIFFERENT value, not a typo of the same one."""
    meta = (report or {}).get("metadata", {})
    unsafe = {"email", "phone", "postcode", "identifier"}
    targets = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        semantic = meta.get(col, {}).get("semantic_type")
        if semantic in unsafe:
            continue
        if semantic == "person_name" or "name" in col.lower():
            targets.append(col)
    return targets


def find_fuzzy_duplicates(df: pd.DataFrame,
                          columns: list[str] | None = None,
                          report: dict | None = None,
                          threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Find candidate fuzzy-duplicate VALUE PAIRS within name-like columns.

    Returns candidates for human review — this function never modifies data.

    Performance note: this is O(n^2) per column (every value compared with
    every other), which does not scale to very large datasets. It is capped at
    MAX_COMPARISONS_PER_COLUMN unique values per column; above that, the check
    is skipped for that column and reported as skipped (not silently ignored).
    A production system with larger data would use blocking/indexing (e.g.
    phonetic keys) to avoid full pairwise comparison — noted as future work.
    """
    target_cols = columns or _fuzzy_candidate_columns(df, report)
    results: dict[str, dict] = {}

    for col in target_cols:
        if col not in df.columns:
            continue
        values = sorted({str(v).strip() for v in df[col].dropna().unique()})

        if len(values) > MAX_COMPARISONS_PER_COLUMN:
            results[col] = {"status": "skipped",
                            "reason": f"{len(values)} unique values exceeds "
                                      f"safety cap of {MAX_COMPARISONS_PER_COLUMN}"}
            continue

        pairs = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                a, b = values[i], values[j]
                if a.lower() == b.lower():
                    continue    # exact/case dupes are deduplication.py's job, not this
                score = _similarity(a.lower(), b.lower())
                if score >= threshold:
                    pairs.append({"value_a": a, "value_b": b,
                                  "similarity": round(score, 3)})

        if pairs:
            pairs.sort(key=lambda p: -p["similarity"])
            results[col] = {
                "status": "ok",
                "candidate_pairs": len(pairs),
                "examples": pairs[:MAX_PAIRS_REPORTED],
                "note": "candidates for human review — NOT auto-merged",
            }

    return {
        "threshold_used": threshold,
        "columns_checked": target_cols,
        "findings": results,
        "total_candidate_pairs": sum(
            r.get("candidate_pairs", 0) for r in results.values() if r.get("status") == "ok"),
    }