"""
MODULE 4 · SCHEMAS — INPUT / OUTPUT CONTRACTS
=============================================
Defines and enforces the contracts between pipeline stages.

WHY: each module hands a file to the next (the "file contract" design). If a
stage silently emits a different shape, the next stage fails in a confusing
place. Validating against a schema makes the failure explicit and immediate.

CONTRACTS
  INPUT   raw CSV            -> must be readable, non-empty, have columns
  STAGE 1 profiling_report.json  -> dataset, metadata, profiling, rules
  STAGE 2 cleaned_data.csv + cleaning_log.json -> quality, steps, row counts
  STAGE 3 validation_report.json -> rule_validation, anomaly_detection, health
  OUTPUT  pipeline_summary.json  -> verdict + paths to every artefact

Public entry points:
  validate_input(df)                 -> raises SchemaError if unusable
  validate_stage_output(stage, data) -> raises SchemaError if contract broken
  describe_schemas()                 -> dict, for documentation
"""
from __future__ import annotations

import pandas as pd


class SchemaError(ValueError):
    """Raised when data does not meet the contract for a pipeline stage."""


# ---- required top-level keys for each stage's output -------------------------
STAGE_SCHEMAS: dict[str, dict] = {
    "profiling": {
        "file": "profiling_report.json",
        "required_keys": ["dataset", "generated_at", "metadata", "profiling", "rules"],
        "nested": {"rules": ["potential_pii", "inconsistent_formats",
                             "suspicious_columns"]},
    },
    "cleaning": {
        "file": "cleaning_log.json",
        "required_keys": ["generated_at", "rows_in", "rows_out", "quality", "steps"],
        "nested": {"quality": ["score_before", "score_after", "delta"]},
    },
    "validation": {
        "file": "validation_report.json",
        "required_keys": ["dataset", "generated_at", "rules_applied",
                          "rule_validation", "anomaly_detection", "health"],
        "nested": {"health": ["health_score", "verdict"]},
    },
    "summary": {
        "file": "pipeline_summary.json",
        "required_keys": ["input", "rows_in", "rows_out", "verdict", "outputs"],
        "nested": {},
    },
}

MIN_ROWS = 1


def validate_input(df: pd.DataFrame, source: str = "input") -> None:
    """Check the raw dataset is usable before the pipeline starts.

    Fails fast with a clear message rather than letting an empty or malformed
    file produce confusing errors three stages later.
    """
    if df is None:
        raise SchemaError(f"{source}: no data was loaded")
    if df.shape[1] == 0:
        raise SchemaError(f"{source}: file has no columns")
    if len(df) < MIN_ROWS:
        raise SchemaError(f"{source}: file has no rows")
    if any(str(c).strip() == "" for c in df.columns):
        raise SchemaError(f"{source}: one or more columns have no name")
    duplicates = [c for c in df.columns if list(df.columns).count(c) > 1]
    if duplicates:
        raise SchemaError(f"{source}: duplicate column names: {sorted(set(duplicates))}")


def validate_stage_output(stage: str, data: dict) -> None:
    """Check a stage produced the structure the next stage depends on."""
    schema = STAGE_SCHEMAS.get(stage)
    if schema is None:
        raise SchemaError(f"unknown pipeline stage: {stage}")
    if not isinstance(data, dict):
        raise SchemaError(f"{stage}: expected a dict, got {type(data).__name__}")

    missing = [k for k in schema["required_keys"] if k not in data]
    if missing:
        raise SchemaError(f"{stage} ({schema['file']}): missing keys {missing}")

    for parent, children in schema.get("nested", {}).items():
        section = data.get(parent, {})
        if not isinstance(section, dict):
            raise SchemaError(f"{stage}: '{parent}' should be a dict")
        missing_children = [k for k in children if k not in section]
        if missing_children:
            raise SchemaError(
                f"{stage} ({schema['file']}): '{parent}' missing {missing_children}")


def describe_schemas() -> dict:
    """Machine-readable description of every contract (used in the docs)."""
    return {
        "input": {
            "format": "CSV",
            "requirements": ["at least 1 row", "at least 1 named column",
                             "no duplicate column names"],
            "note": "read with dtype=str so identifiers keep leading zeros",
        },
        "stages": STAGE_SCHEMAS,
    }