"""
MODULE 1 · FILE 5 — PROFILING API (ORCHESTRATOR)
================================================
The public entry point for Module 1. Wires together Files 1-4 and writes the
single deliverable: profiling_report.json.

  Input :  a CSV path
  Output:  outputs/profiling_report.json  (+ figures in outputs/figures/)

Pipeline:
  metadata (File 1) -> profiling (File 2) -> rules (File 3) -> figures (File 4)

Usage (CLI):
  python profile.py --input data/raw/broadband_customers.csv
  python profile.py --input data/raw/any.csv --no-figures

Usage (import):
  from profiling_api import build_profiling_report
  report = build_profiling_report(df)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))          # sibling files (metadata, profiling_engine, ...)
sys.path.append(str(HERE.parent))   # common.py

from metadata import extract_metadata          # noqa: E402
from profiling_engine import profile_data       # noqa: E402
from rule_engine import run_rules               # noqa: E402
from visualisations import generate_all         # noqa: E402

# default output location (falls back if common.py isn't importable)
try:
    from common import OUTPUTS_DIR              # noqa: E402
except Exception:
    OUTPUTS_DIR = HERE.parents[1] / "outputs"


def build_profiling_report(df: pd.DataFrame, source: str = "",
                           with_figures: bool = True,
                           figures_dir: str | Path | None = None) -> dict:
    """Run the full Module 1 pipeline and return the profiling report dict."""
    report = {
        "dataset": {
            "source": source,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": extract_metadata(df),   # File 1
        "profiling": profile_data(df),      # File 2
        "rules": run_rules(df),             # File 3
        "figures": {},
    }
    if with_figures:
        fig_dir = figures_dir or (OUTPUTS_DIR / "figures")
        report["figures"] = generate_all(df, fig_dir)   # File 4
    return report


def main():
    parser = argparse.ArgumentParser(description="Module 1 — Data Profiling API")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "profiling_report.json"),
                        help="Where to write the JSON report")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip generating PNG visualisations")
    args = parser.parse_args()

    input_path = Path(args.input)
    print(f"[Module 1] Profiling {input_path.name}")
    df = pd.read_csv(input_path)

    report = build_profiling_report(
        df, source=str(input_path), with_figures=not args.no_figures
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    r = report
    print(f"  rows={r['dataset']['rows']} cols={r['dataset']['columns']}")
    print(f"  PII flagged: {list(r['rules']['potential_pii'].keys())}")
    print(f"  suspicious columns: {list(r['rules']['suspicious_columns'].keys())}")
    print(f"  ✔ report written to {out_path}")


if __name__ == "__main__":
    main()