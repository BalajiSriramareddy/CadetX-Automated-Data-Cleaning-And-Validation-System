"""
MODULE 3 · FILE 5 — VALIDATION API (ORCHESTRATOR)
=================================================
The public entry point for Module 3. Chains the validation steps and writes the
single deliverable: validation_report.json.

  Input :  data/processed/cleaned_data.csv
           (optional) outputs/profiling_report.json  — improves rule inference
           (optional) rules.yaml                     — adds business rules
  Output:  outputs/validation_report.json

Pipeline:
  build_rules (config.py)        -> what to check   [inferred + declared]
  validate_rules (rule_validation) -> rule violations
  detect_anomalies (anomaly_detection) -> statistical + ML anomalies
  compute_health_score (health_score)  -> 0-100 score + PASS/WARNING/FAIL verdict

Usage (CLI):
  python validate.py --input data/processed/cleaned_data.csv
  python validate.py --input any.csv --report outputs/profiling_report.json \
                     --rules rules.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

from config import build_rules                    # noqa: E402
from rule_validation import validate_rules        # noqa: E402
from anomaly_detection import detect_anomalies    # noqa: E402
from health_score import compute_health_score     # noqa: E402

try:
    from common import OUTPUTS_DIR, CLEANED_DATA, VALIDATION_REPORT  # noqa: E402
except Exception:
    OUTPUTS_DIR = HERE.parents[1] / "outputs"
    CLEANED_DATA = HERE.parents[1] / "data" / "processed" / "cleaned_data.csv"
    VALIDATION_REPORT = OUTPUTS_DIR / "validation_report.json"


def build_validation_report(df: pd.DataFrame,
                            report: dict | None = None,
                            rules_path: str | Path | None = None,
                            contamination: float = 0.05,
                            source: str = "") -> dict:
    """Run the full Module 3 pipeline and return the validation report dict."""
    rules = build_rules(df, report, yaml_path=rules_path)
    rule_results = validate_rules(df, rules)
    anomaly_results = detect_anomalies(df, contamination=contamination)
    health = compute_health_score(rule_results, anomaly_results, len(df))

    return {
        "dataset": {
            "source": source,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rules_applied": {
            "columns_with_rules": list(rules.keys()),
            "rules": rules,
            "declared_rules_file": str(rules_path) if rules_path else None,
        },
        "rule_validation": rule_results,
        "anomaly_detection": anomaly_results,
        "health": health,
    }


def main():
    parser = argparse.ArgumentParser(description="Module 3 — Data Validation API")
    parser.add_argument("--input", default=str(CLEANED_DATA),
                        help="CSV to validate (default: cleaned_data.csv)")
    parser.add_argument("--report", help="Optional profiling_report.json from Module 1")
    parser.add_argument("--rules", help="Optional rules.yaml with business rules")
    parser.add_argument("--output", default=str(VALIDATION_REPORT))
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly proportion for Isolation Forest")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    profiling = None
    if args.report and Path(args.report).exists():
        with open(args.report, encoding="utf-8") as f:
            profiling = json.load(f)

    print(f"[Module 3] Validating {input_path.name}")
    result = build_validation_report(df, profiling, args.rules,
                                     args.contamination, str(input_path))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    h = result["health"]
    rv = result["rule_validation"]["summary"]
    print(f"  rules applied to {len(result['rules_applied']['columns_with_rules'])} columns")
    print(f"  rule violations: {rv['total_violations']}"
          f" across {len(rv['columns_with_violations'])} columns")
    print(f"  health score: {h['health_score']}  ->  {h['verdict']}")
    print(f"  ✔ report written to {out_path}")


if __name__ == "__main__":
    main()