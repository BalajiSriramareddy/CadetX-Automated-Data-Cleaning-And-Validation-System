"""Shared helpers used across modules: JSON read/write and project paths.

Keeping these in one place means every module reads and writes its
contract files the same way.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Project root = two levels up from this file (modules/common.py -> project root)
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# The four contract files. Every module reads/writes by these names.
PROFILING_REPORT = OUTPUTS_DIR / "profiling_report.json"
CLEANED_DATA = PROCESSED_DIR / "cleaned_data.csv"
CLEANING_LOG = OUTPUTS_DIR / "cleaning_log.json"
VALIDATION_REPORT = OUTPUTS_DIR / "validation_report.json"


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dict to JSON, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✔ wrote {path.relative_to(ROOT)}")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON contract file produced by an earlier module."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
