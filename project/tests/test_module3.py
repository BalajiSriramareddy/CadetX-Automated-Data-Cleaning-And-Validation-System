"""
MODULE 3 — TEST SUITE
=====================
Known-answer tests for the validation module.

Run:  pytest tests/test_module3.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "modules" / "m3_validation"))

from rule_validation import validate_rules
from config import infer_rules, merge_rules, _normalise_declared
from anomaly_detection import iqr_outliers, isolation_forest_anomalies
from health_score import compute_health_score
from validate import build_validation_report


# ---------------- rule_validation ----------------

def test_format_violations_counted():
    df = pd.DataFrame({"email": ["a@x.com", "bad-email", "c@y.com"]})
    result = validate_rules(df, {"email": {"format": "email"}})
    assert result["column_violations"]["email"][0]["violations"] == 1


def test_range_violations_counted():
    df = pd.DataFrame({"nps": [8, 11, -2, 5]})
    result = validate_rules(df, {"nps": {"range": {"min": 0, "max": 10}}})
    assert result["column_violations"]["nps"][0]["violations"] == 2


def test_allowed_values_violation():
    df = pd.DataFrame({"churn": ["Yes", "No", "Maybe"]})
    result = validate_rules(df, {"churn": {"allowed": ["Yes", "No"]}})
    assert result["column_violations"]["churn"][0]["violations"] == 1


def test_clean_data_has_no_violations():
    df = pd.DataFrame({"nps": [1, 5, 10]})
    result = validate_rules(df, {"nps": {"range": {"min": 0, "max": 10}}})
    assert result["summary"]["total_violations"] == 0


# ---------------- config (the two layers) ----------------

def test_layer1_infers_rules_without_config():
    """Zero-config: rules are still inferred from the profiling report."""
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"]})
    report = {"metadata": {"email": {"semantic_type": "email"}}}
    rules = infer_rules(df, report)
    assert rules["email"]["format"] == "email"


def test_declared_rules_override_inferred():
    inferred = {"nps": {"range": {"min": 0, "max": 100}}}
    declared = {"nps": {"range": {"min": 0, "max": 10}}}
    merged = merge_rules(inferred, declared)
    assert merged["nps"]["range"]["max"] == 10


def test_yaml_boolean_coercion_guarded():
    """The 'Norway problem': YAML turns Yes/No into booleans — must be coerced back."""
    spec = _normalise_declared({"allowed": [True, False]})
    assert spec["allowed"] == ["Yes", "No"]


# ---------------- anomaly_detection ----------------

def test_iqr_detects_extreme_value():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 12, 500]})
    result = iqr_outliers(df)
    assert result["x"]["outliers"] == 1


def test_isolation_forest_skips_tiny_dataset():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = isolation_forest_anomalies(df)
    assert result["status"] != "ok"


def test_isolation_forest_runs_on_valid_data():
    np.random.seed(0)
    df = pd.DataFrame({"a": np.random.normal(0, 1, 100),
                       "b": np.random.normal(5, 2, 100)})
    result = isolation_forest_anomalies(df, contamination=0.05)
    assert result["status"] == "ok"
    assert result["anomalies_found"] > 0


# ---------------- health_score ----------------

def test_clean_data_scores_pass():
    result = compute_health_score({"column_violations": {}}, None, 100)
    assert result["health_score"] == 100.0
    assert result["verdict"] == "PASS"


def test_critical_violations_score_worse_than_low():
    """Severity weighting: a required-field failure must hurt more than an outlier."""
    critical = compute_health_score(
        {"column_violations": {"id": [{"rule": "required", "violations": 50}]}}, None, 100)
    minor = compute_health_score(
        {"column_violations": {"x": [{"rule": "range:[0,10]", "violations": 50}]}}, None, 100)
    assert critical["health_score"] < minor["health_score"]


def test_broken_data_fails():
    result = compute_health_score(
        {"column_violations": {"id": [{"rule": "required", "violations": 90}]}}, None, 100)
    assert result["verdict"] == "FAIL"


# ---------------- integration ----------------

def test_validation_report_has_required_keys():
    df = pd.DataFrame({"nps": [1, 5, 10], "email": ["a@x.com", "b@y.com", "c@z.com"]})
    report = build_validation_report(df)
    for key in ["dataset", "generated_at", "rules_applied",
                "rule_validation", "anomaly_detection", "health"]:
        assert key in report


def test_report_works_with_no_config_at_all():
    """Dataset-agnostic: no profiling report, no rules.yaml — still validates."""
    df = pd.DataFrame({"anything": [1, 2, 3], "text": ["a", "b", "c"]})
    report = build_validation_report(df)
    assert report["health"]["verdict"] in ("PASS", "WARNING", "FAIL")