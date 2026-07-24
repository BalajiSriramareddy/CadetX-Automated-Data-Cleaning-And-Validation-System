"""
MODULE 3 · FILE 2 — RULES CONFIGURATION (THE TWO-LAYER ENGINE)
==============================================================
Decides WHAT rules to apply. This is what makes the claim
"validates any dataset with zero code changes" actually true.

TWO LAYERS
  Layer 1 — INFERRED (always runs, no config needed)
      Builds sensible default rules from Module 1's profiling report:
        semantic_type "email"    -> format: email
        semantic_type "phone"    -> format: uk_phone
        semantic_type "postcode" -> format: uk_postcode
        low-cardinality category -> allowed: the values actually observed
      So pointing the system at ANY csv still produces real validation.

  Layer 2 — DECLARED (optional rules.yaml)
      Business rules that CANNOT be inferred from data, e.g. "nps_score must be
      0-10". A dataset with no yaml still validates; a yaml adds domain rules.

  Merge policy: DECLARED wins. An explicit human rule always overrides an
  inferred guess.

This is the honest answer to "how does it know NPS is 0-10?" — it doesn't, and
it can't. That is a business rule, so it lives in config, not in code.

Public entry point:  build_rules(df, report=None, yaml_path=None) -> dict
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:  # pyyaml is optional — inferred layer still works without it
    yaml = None

# a category column must be at least this repetitive before we infer an
# "allowed values" rule from it (otherwise we'd freeze free text into a rule)
MAX_CATEGORY_UNIQUE = 20
MAX_CATEGORY_RATIO = 0.05

SEMANTIC_TO_FORMAT = {
    "email": "email",
    "phone": "uk_phone",
    "postcode": "uk_postcode",
}


def infer_rules(df: pd.DataFrame, report: dict | None = None) -> dict:
    """LAYER 1 — build default rules from the profiling report / the data itself."""
    rules: dict[str, dict] = {}
    meta = (report or {}).get("metadata", {})
    n = len(df)

    for col in df.columns:
        info = meta.get(col, {})
        semantic = info.get("semantic_type")
        inferred_type = info.get("inferred_type")
        spec: dict = {}

        # format rules from semantic type
        if semantic in SEMANTIC_TO_FORMAT:
            spec["format"] = SEMANTIC_TO_FORMAT[semantic]

        # allowed-values rule for genuinely categorical columns
        is_categorical = (
            inferred_type == "categorical"
            or (inferred_type is None
                and not pd.api.types.is_numeric_dtype(df[col])
                and df[col].nunique(dropna=True) <= max(MAX_CATEGORY_UNIQUE,
                                                        MAX_CATEGORY_RATIO * n))
        )
        if is_categorical and semantic not in ("email", "phone", "postcode",
                                               "identifier", "person_name"):
            observed = sorted({str(v) for v in df[col].dropna().unique()})
            if 0 < len(observed) <= MAX_CATEGORY_UNIQUE:
                spec["allowed"] = observed

        if spec:
            rules[col] = spec

    return rules


def _normalise_declared(spec: dict) -> dict:
    """Guard against YAML's boolean coercion (the 'Norway problem').

    YAML 1.1 turns unquoted Yes/No/On/Off/True/False into booleans, so
    `allowed: [Yes, No]` arrives as [True, False] and every valid row fails.
    We coerce booleans in an allowed-list back to "Yes"/"No" strings.
    """
    spec = dict(spec or {})
    if "allowed" in spec and isinstance(spec["allowed"], list):
        spec["allowed"] = [
            ("Yes" if v is True else "No" if v is False else v)
            for v in spec["allowed"]
        ]
    return spec


def load_yaml_rules(yaml_path: str | Path | None) -> dict:
    """LAYER 2 — load declared business rules from a rules.yaml, if provided."""
    if not yaml_path:
        return {}
    path = Path(yaml_path)
    if not path.exists():
        return {}
    if yaml is None:
        raise ImportError("pyyaml is required to read rules.yaml (pip install pyyaml)")
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    columns = loaded.get("columns", loaded)
    return {col: _normalise_declared(spec) for col, spec in (columns or {}).items()}


def merge_rules(inferred: dict, declared: dict) -> dict:
    """Merge the two layers. DECLARED wins on conflict, key by key."""
    merged = {col: dict(spec) for col, spec in inferred.items()}
    for col, spec in (declared or {}).items():
        merged.setdefault(col, {})
        merged[col].update(spec or {})   # explicit human rule overrides inference
    return merged


def build_rules(df: pd.DataFrame, report: dict | None = None,
                yaml_path: str | Path | None = None) -> dict:
    """Build the final rules spec: inferred defaults + optional declared rules."""
    inferred = infer_rules(df, report)
    declared = load_yaml_rules(yaml_path)
    return merge_rules(inferred, declared)