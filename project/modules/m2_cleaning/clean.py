"""
MODULE 2 — DATA CLEANING
========================
Fix and improve the dataset after profiling.

FILE CONTRACT
  Input :  data/raw/<dataset>.csv  +  outputs/profiling_report.json
  Output:  data/processed/cleaned_data.csv
           outputs/cleaning_log.json   (what changed + before/after quality score)

WHAT TO BUILD (Week 2)
  - impute missing values       (start statistical: mean/median/mode; ML optional later)
  - detect & remove duplicates
  - normalise formats           (dates, strings, categories)
  - data-quality score          (compute BEFORE and AFTER, report the delta)

The before/after quality delta is the line that sells this module to a
recruiter — it's a quantified result. Make it real.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (write_json, read_json, CLEANED_DATA, CLEANING_LOG,
                    PROFILING_REPORT, RAW_DIR)  # noqa: E402


def quality_score(df: pd.DataFrame) -> float:
    """A simple 0-100 dataset health score.

    Starter definition: % of cells that are non-missing and non-duplicate.
    TODO: refine — weight by column importance, add format-validity, etc.
    """
    completeness = 1 - df.isna().sum().sum() / df.size
    uniqueness = 1 - df.duplicated().sum() / len(df)
    return round(100 * (0.7 * completeness + 0.3 * uniqueness), 2)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: fill missing values. Numeric -> median, categorical -> mode."""
    raise NotImplementedError


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: remove exact (and later fuzzy) duplicate rows."""
    raise NotImplementedError


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: standardise dates, trim/lowercase strings, unify categories."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Module 2 — Data Cleaning")
    parser.add_argument("--input", required=False)
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else sorted(RAW_DIR.glob("*.csv"))[0]
    df = pd.read_csv(input_path)
    profile = read_json(PROFILING_REPORT)  # use M1's findings to guide cleaning

    score_before = quality_score(df)

    # ---- Week 2: implement these ----
    # df = impute_missing(df)
    # df = drop_duplicates(df)
    # df = normalise(df)

    score_after = quality_score(df)

    CLEANED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA, index=False)
    write_json(CLEANING_LOG, {
        "quality_score_before": score_before,
        "quality_score_after": score_after,
        "quality_delta": round(score_after - score_before, 2),
        "actions": ["TODO: log each cleaning step taken"],
        "used_profile_flags": profile.get("flags", {}),
    })
    print("[M2] Done.")


if __name__ == "__main__":
    main()
