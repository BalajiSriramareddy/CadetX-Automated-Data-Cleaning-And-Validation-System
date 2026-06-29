"""
MODULE 1 — DATA PROFILING
=========================
Understand the dataset *before* any cleaning happens.

FILE CONTRACT
  Input :  a raw CSV  (data/raw/<your_dataset>.csv)
  Output:  outputs/profiling_report.json

WHAT GOES IN THE REPORT
  - row / column counts
  - inferred column types
  - missing-value counts and % per column
  - basic distribution stats (min/max/mean/median for numeric)
  - cardinality (unique values) per column
  - correlation between numeric columns
  - flags: suspicious columns, potential PII (email/phone/id-like)

This is the analyst-strength module — lead with this one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import write_json, PROFILING_REPORT, RAW_DIR  # noqa: E402


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(path)


def basic_shape(df: pd.DataFrame) -> dict:
    """Row/column counts and column list."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
    }


def infer_types(df: pd.DataFrame) -> dict:
    """Inferred dtype per column."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def missing_values(df: pd.DataFrame) -> dict:
    """Missing count and percentage per column."""
    out = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        out[col] = {
            "missing": n_missing,
            "missing_pct": round(100 * n_missing / len(df), 2),
        }
    return out


def distributions(df: pd.DataFrame) -> dict:
    """Min/max/mean/median/std for numeric columns."""
    num = df.select_dtypes(include="number")
    out = {}
    for col in num.columns:
        out[col] = {
            "min": float(num[col].min()),
            "max": float(num[col].max()),
            "mean": round(float(num[col].mean()), 3),
            "median": float(num[col].median()),
            "std": round(float(num[col].std()), 3),
        }
    return out


def cardinality(df: pd.DataFrame) -> dict:
    """Unique value count per column."""
    return {col: int(df[col].nunique()) for col in df.columns}


def correlations(df: pd.DataFrame) -> dict:
    """Pairwise correlation between numeric columns."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {}
    return num.corr().round(3).to_dict()


def flag_pii(df: pd.DataFrame) -> list[str]:
    """Flag columns whose NAME suggests personal/sensitive data.

    TODO: upgrade from name-matching to value-pattern checks
          (regex for emails, phone numbers) for a stronger flag.
    """
    pii_hints = ("email", "phone", "name", "address", "postcode",
                 "ssn", "id", "dob", "birth")
    return [c for c in df.columns if any(h in c.lower() for h in pii_hints)]


def build_report(df: pd.DataFrame) -> dict:
    """Assemble the full profiling_report.json contract."""
    return {
        "shape": basic_shape(df),
        "types": infer_types(df),
        "missing_values": missing_values(df),
        "distributions": distributions(df),
        "cardinality": cardinality(df),
        "correlations": correlations(df),
        "flags": {"potential_pii_columns": flag_pii(df)},
    }


def main():
    parser = argparse.ArgumentParser(description="Module 1 — Data Profiling")
    parser.add_argument("--input", required=False,
                        help="Path to raw CSV (default: first CSV in data/raw/)")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    else:
        csvs = sorted(RAW_DIR.glob("*.csv"))
        if not csvs:
            raise SystemExit("No CSV found in data/raw/. Pass --input <path>.")
        input_path = csvs[0]

    print(f"[M1] Profiling {input_path}")
    df = load_data(input_path)
    report = build_report(df)
    write_json(PROFILING_REPORT, report)
    print("[M1] Done.")


if __name__ == "__main__":
    main()
