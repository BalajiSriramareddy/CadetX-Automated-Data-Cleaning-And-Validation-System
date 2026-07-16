"""
MODULE 2 · FILE 5 — CLEANING API (ORCHESTRATOR)
===============================================
The public entry point for Module 2. Chains the cleaning steps and writes the
deliverables.

  Input :  a CSV path  (+ optional profiling_report.json from Module 1)
  Output:  data/processed/cleaned_data.csv
           outputs/cleaning_log.json   (every action + before/after quality score)

Pipeline ORDER matters:
  1. deduplicate   - remove duplicate rows first
  2. normalise     - fix types/dates/categories (£ -> numeric) BEFORE imputing,
                     so imputation sees real numbers and uses the median
  3. impute        - fill remaining missing values by type
  4. score         - measure quality before vs after to prove it worked

Usage (CLI):
  python clean.py --input data/raw/broadband_customers.csv
  python clean.py --input data/raw/any.csv --report outputs/profiling_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))          # sibling files
sys.path.append(str(HERE.parent))   # common.py

from deduplication import remove_duplicates   # noqa: E402
from normalisation import normalise            # noqa: E402
from imputation import impute_missing          # noqa: E402
from quality_score import score_before_after   # noqa: E402

try:
    from common import PROCESSED_DIR, OUTPUTS_DIR, CLEANED_DATA, CLEANING_LOG  # noqa: E402
except Exception:
    PROCESSED_DIR = HERE.parents[1] / "data" / "processed"
    OUTPUTS_DIR = HERE.parents[1] / "outputs"
    CLEANED_DATA = PROCESSED_DIR / "cleaned_data.csv"
    CLEANING_LOG = OUTPUTS_DIR / "cleaning_log.json"


def clean_dataset(df: pd.DataFrame, report: dict | None = None):
    """Run the full cleaning pipeline. Returns (cleaned_df, log_dict)."""
    before = df.copy()

    df, dedup_log = remove_duplicates(df, report)
    df, norm_log = normalise(df, report)
    df, impute_log = impute_missing(df, report)

    quality = score_before_after(before, df)

    log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_in": int(len(before)),
        "rows_out": int(len(df)),
        "quality": quality,
        "steps": {
            "deduplication": dedup_log,
            "normalisation": norm_log,
            "imputation": impute_log,
        },
    }
    return df, log


def main():
    parser = argparse.ArgumentParser(description="Module 2 — Data Cleaning API")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--report", help="Optional profiling_report.json from Module 1")
    parser.add_argument("--output", default=str(CLEANED_DATA))
    parser.add_argument("--log", default=str(CLEANING_LOG))
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    report = None
    if args.report and Path(args.report).exists():
        with open(args.report, encoding="utf-8") as f:
            report = json.load(f)

    print(f"[Module 2] Cleaning {Path(args.input).name}")
    cleaned, log = clean_dataset(df, report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    q = log["quality"]
    print(f"  rows: {log['rows_in']} -> {log['rows_out']}")
    print(f"  quality score: {q['score_before']} -> {q['score_after']}  (+{q['delta']})")
    print(f"  ✔ cleaned data -> {out_path}")
    print(f"  ✔ cleaning log -> {log_path}")


if __name__ == "__main__":
    main()