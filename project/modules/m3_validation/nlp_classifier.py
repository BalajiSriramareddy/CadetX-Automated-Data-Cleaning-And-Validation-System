"""
MODULE 3 · NLP COLUMN CLASSIFIER
================================
Classifies what a column MEANS from its name and values, and flags when a
column's NAME disagrees with its ACTUAL content — a "mislabelled column".

WHY "NLP-lite" rather than a trained language model: classifying a column
name like "cust_email" as meaning "email" doesn't need a transformer model —
keyword/token matching against a curated vocabulary is fast, needs no training
data, and — critically — is fully explainable (you can always say exactly why
a column was classified a given way). A full NLP model here would be
over-engineering: hard to justify, hard to explain, and no more accurate than
this for the vocabulary size involved. This is intentionally a text
CLASSIFICATION step (tokenising the name into a semantic category), which is
where "NLP column classifier" earns its name, not a marketing label.

Three jobs (from the brief):
  1. Classify column meaning        -> classify_column_meaning()
  2. Detect mislabeled columns      -> detect_mislabeled_columns()
  3. Detect semantic inconsistencies -> detect_semantic_inconsistencies()

Public entry point:  classify_columns(df, report=None) -> dict
"""
from __future__ import annotations

import re

import pandas as pd

# vocabulary: semantic category -> tokens that suggest it in a column NAME
NAME_VOCAB = {
    "identifier": ["id", "identifier", "code", "no", "number", "ref", "key"],
    "email": ["email", "mail"],
    "phone": ["phone", "mobile", "tel", "contact"],
    "postcode": ["postcode", "zip", "postal"],
    "person_name": ["name", "customer", "client", "firstname", "lastname", "surname"],
    "date": ["date", "time", "created", "updated", "dob", "birth", "timestamp"],
    "money": ["price", "cost", "charge", "fee", "amount", "revenue", "salary", "spend"],
    "quantity": ["count", "qty", "quantity", "number_of", "num"],
    "score": ["score", "rating", "rank", "nps", "satisfaction"],
    "flag": ["is_", "has_", "flag", "active", "enabled"],
    "geography": ["region", "country", "city", "state", "county", "location", "area"],
    "category": ["type", "category", "segment", "class", "tier", "plan", "package"],
}

# what VALUE evidence would support each category (used to cross-check the name)
VALUE_CHECKS = {
    "email": lambda s: s.str.contains("@", na=False).mean() >= 0.8,
    "money": lambda s: pd.to_numeric(
        s.str.replace(r"[£$€,]", "", regex=True), errors="coerce").notna().mean() >= 0.8,
    "score": lambda s: pd.to_numeric(s, errors="coerce").notna().mean() >= 0.8,
    "quantity": lambda s: pd.to_numeric(s, errors="coerce").notna().mean() >= 0.8,
    "date": lambda s: pd.to_datetime(s, errors="coerce", format="mixed").notna().mean() >= 0.7,
    "flag": lambda s: s.str.lower().isin(
        ["yes", "no", "true", "false", "0", "1", "y", "n"]).mean() >= 0.9,
}

_TOKEN_SPLIT_RE = re.compile(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])")


def _tokenise(column_name: str) -> list[str]:
    """Split a column name into lowercase tokens: 'customerEmail' / 'customer_email'
    / 'Customer-Email' all -> ['customer', 'email']. This is the "NLP" step:
    treating the column name as text to be tokenised and matched, not a literal
    string to search inside."""
    return [t.lower() for t in _TOKEN_SPLIT_RE.split(column_name) if t]


def classify_column_meaning(column_name: str) -> dict:
    """Classify a column's likely meaning from its NAME tokens.
    Returns the best-matching category and the tokens that triggered it."""
    tokens = _tokenise(column_name)
    matches = {}
    for category, vocab in NAME_VOCAB.items():
        hit_tokens = [t for t in tokens if any(v in t for v in vocab)]
        if hit_tokens:
            matches[category] = hit_tokens
    if not matches:
        return {"predicted_meaning": "unknown", "confidence": 0.0, "matched_tokens": []}
    # prefer the category with the most matched tokens (a weak but explainable score)
    best = max(matches, key=lambda c: len(matches[c]))
    confidence = min(1.0, 0.5 + 0.25 * len(matches[best]))
    return {"predicted_meaning": best, "confidence": confidence,
            "matched_tokens": matches[best]}


def detect_mislabeled_columns(df: pd.DataFrame) -> dict:
    """Cross-check the NAME-based classification against the VALUES.

    A column is flagged 'mislabelled' when its name strongly suggests a
    category (e.g. 'email') but its values don't support that at all — e.g. a
    column called 'customer_email' that actually holds Yes/No values.
    """
    findings = {}
    for col in df.columns:
        meaning = classify_column_meaning(col)
        category = meaning["predicted_meaning"]
        if category not in VALUE_CHECKS or meaning["confidence"] < 0.5:
            continue  # no strong enough name signal to cross-check
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue
        try:
            value_supports = bool(VALUE_CHECKS[category](sample))
        except Exception:
            value_supports = False
        if not value_supports:
            findings[col] = {
                "name_suggests": category,
                "matched_tokens": meaning["matched_tokens"],
                "issue": "column values do not support the meaning suggested by its name",
                "example_values": sample.unique()[:5].tolist(),
            }
    return findings


def detect_semantic_inconsistencies(df: pd.DataFrame) -> dict:
    """Find columns whose meaning shifts partway through — a single column
    holding two conceptually different kinds of value.

    Implemented as: does the column's value TYPE (number vs date vs plain
    text) vary in a way not explained by simple formatting? This overlaps in
    spirit with Module 1's mixed-type detector, but classifies the semantic
    CATEGORY of each value-group rather than just flagging "mixed types" —
    e.g. distinguishing "some values are dates, some are plain numbers" (a
    real semantic inconsistency) from "some values have £ signs" (a format
    issue, not a meaning issue).
    """
    findings = {}
    for col in df.columns:
        sample = df[col].dropna().astype(str).str.strip()
        if len(sample) < 5:
            continue

        looks_money = sample.str.match(r"^[£$€]?\d+(\.\d+)?$")
        looks_date = pd.to_datetime(sample, errors="coerce", format="mixed").notna()
        looks_word = sample.str.match(r"^[A-Za-z\s]+$")

        groups_present = sum([
            looks_money.mean() > 0.15,
            (looks_date & ~looks_money).mean() > 0.15,
            looks_word.mean() > 0.15,
        ])
        if groups_present >= 2:
            findings[col] = {
                "issue": "column mixes conceptually different value categories",
                "money_like_pct": round(100 * float(looks_money.mean()), 1),
                "date_like_pct": round(100 * float(looks_date.mean()), 1),
                "word_like_pct": round(100 * float(looks_word.mean()), 1),
            }
    return findings


def classify_columns(df: pd.DataFrame, report: dict | None = None) -> dict:
    """Run all three NLP classifier jobs and return one dict."""
    meanings = {col: classify_column_meaning(col) for col in df.columns}
    return {
        "column_meanings": meanings,
        "mislabeled_columns": detect_mislabeled_columns(df),
        "semantic_inconsistencies": detect_semantic_inconsistencies(df),
    }
