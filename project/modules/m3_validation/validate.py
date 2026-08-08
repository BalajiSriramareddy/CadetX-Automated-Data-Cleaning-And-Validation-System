"""
MODULE 3 · FILE 5 — VALIDATION API (ORCHESTRATOR)
==================================================
Chains validation steps and writes validation_report.json.

CORE pipeline (always runs):
  1. build_rules   -> inferred defaults + optional rules.yaml
  2. validate_rules -> rule violations
  3. detect_anomalies -> IQR + Isolation Forest
  4. compute_health_score -> 0-100 score + PASS/WARNING/FAIL verdict

OPTIONAL advanced steps (config-controlled, OFF by default):
  - Local Outlier Factor / autoencoder  (additional anomaly methods)
  - NLP column classifier                (meaning, mislabelled columns)
  - AI error detection                   (impossible values, patterns, drift)
  - evaluation                           (only runs if ground-truth anomaly
                                          row indices are supplied — see
                                          evaluation.py for why this can't
                                          run unconditionally)

Kept optional because they cost more to run and are harder to explain than
the core; enable them for a specific, justified need.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

from config import build_rules
from rule_validation import validate_rules
from anomaly_detection import detect_anomalies
from health_score import compute_health_score

# optional/advanced components
from advanced_anomaly import detect_lof, detect_autoencoder
from nlp_classifier import classify_columns
from ai_error_detection import detect_impossible_values, detect_suspicious_patterns, detect_data_drift
from evaluation import evaluate_against_planted_anomalies


def build_validation_report(df: pd.DataFrame, report: dict | None = None,
                            rules_path: str | Path | None = None,
                            contamination: float = 0.05, source: str = "",
                            options: dict | None = None,
                            reference_df: pd.DataFrame | None = None,
                            true_anomaly_indices: list[int] | None = None) -> dict:
    """Run the validation pipeline. `options` (all default False):
      local_outlier_factor : bool
      autoencoder           : bool
      nlp_classifier        : bool
      ai_error_detection    : bool
      data_drift             : bool -> requires reference_df
      evaluate               : bool -> requires true_anomaly_indices (ground truth)
    """
    options = options or {}

    rules = build_rules(df, report, yaml_path=rules_path)
    rule_results = validate_rules(df, rules)
    anomaly_results = detect_anomalies(df, contamination=contamination)
    health = compute_health_score(rule_results, anomaly_results, len(df))

    result = {
        "dataset": {"source": source, "rows": int(len(df)), "columns": int(df.shape[1])},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rules_applied": {"columns_with_rules": list(rules.keys()), "rules": rules,
                          "declared_rules_file": str(rules_path) if rules_path else None},
        "rule_validation": rule_results,
        "anomaly_detection": anomaly_results,
        "health": health,
        "advanced": {},
    }

    if options.get("local_outlier_factor"):
        result["advanced"]["local_outlier_factor"] = detect_lof(df, contamination)
    if options.get("autoencoder"):
        result["advanced"]["autoencoder"] = detect_autoencoder(df, contamination)
    if options.get("nlp_classifier"):
        result["advanced"]["nlp_classifier"] = classify_columns(df, report)
    if options.get("ai_error_detection"):
        result["advanced"]["ai_error_detection"] = {
            "impossible_values": detect_impossible_values(df, report),
            "suspicious_patterns": detect_suspicious_patterns(df),
        }
    if options.get("data_drift") and reference_df is not None:
        result["advanced"]["data_drift"] = detect_data_drift(reference_df, df)
    if options.get("evaluate") and true_anomaly_indices is not None:
        predicted = anomaly_results.get("multivariate", {}).get("example_row_indices", [])
        result["advanced"]["evaluation"] = evaluate_against_planted_anomalies(
            len(df), true_anomaly_indices, predicted)

    return result
