"""
MODULE 1 · FILE 6b — TEST SUITE
===============================
Known-answer tests for the profiling module. Each test feeds data where we
KNOW the correct answer and asserts the code produces it.

Run:  pytest tests/test_module1.py -v
"""
import sys
from pathlib import Path

import pandas as pd

# make the module importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "modules" / "m1_profiling"))

from metadata import (infer_type, detect_semantic_type, detect_mixed_types,
                      extract_metadata)
from profiling_engine import (missing_value_analysis, cardinality_analysis,
                              profile_data)
from rule_engine import run_rules
from profiling_api import build_profiling_report


# ---------------- metadata.py ----------------

def test_empty_column_infers_empty():
    """The guard clause: an all-missing column returns 'empty'."""
    s = pd.Series([None, None, None])
    assert infer_type(s) == "empty"


def test_numeric_stored_as_text_is_detected():
    """A column of '£' strings should be flagged numeric_as_text."""
    s = pd.Series(["£29.99", "£44.99", "£54.99"])
    assert infer_type(s) == "numeric_as_text"


def test_mixed_type_column_flagged():
    """Numbers mixed with currency strings -> is_mixed True."""
    s = pd.Series([29.99, "£44.99", 54.99])
    assert detect_mixed_types(s)["is_mixed"] is True


def test_email_semantic_detection():
    """A clean email column is detected as email."""
    s = pd.Series([f"user{i}@example.com" for i in range(10)])
    assert detect_semantic_type(s, "contact") == "email"


# ---------------- profiling_engine.py ----------------

def test_missing_value_counts():
    df = pd.DataFrame({"a": [1, None, 3, None]})
    result = missing_value_analysis(df)
    assert result["per_column"]["a"]["missing"] == 2
    assert result["per_column"]["a"]["missing_pct"] == 50.0


def test_constant_column_flagged():
    df = pd.DataFrame({"const": ["X", "X", "X", "X"]})
    result = cardinality_analysis(df)
    assert result["const"]["is_constant"] is True


def test_profile_data_returns_all_sections():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = profile_data(df)
    for key in ["missing_values", "type_consistency", "cardinality",
                "correlations", "distributions"]:
        assert key in result


# ---------------- rule_engine.py ----------------

def test_dirty_email_column_still_flagged_as_pii():
    """Name-hint triggering: a dirty email column is still caught."""
    df = pd.DataFrame({"email": ["a@x.com", "bad", "c@y.com", "nope", "e@z.com"]})
    rules = run_rules(df)
    assert "email" in rules["potential_pii"]


def test_invalid_emails_counted():
    df = pd.DataFrame({"email": ["a@x.com", "bad-email", "c@y.com"]})
    rules = run_rules(df)
    assert rules["inconsistent_formats"]["email"]["invalid_count"] == 1


def test_constant_column_is_suspicious():
    df = pd.DataFrame({"flag": ["Y", "Y", "Y"]})
    rules = run_rules(df)
    assert "flag" in rules["suspicious_columns"]


# ---------------- profile.py (integration) ----------------

def test_report_has_required_keys():
    df = pd.DataFrame({"a": [1, 2, 3], "email": ["x@y.com", "z@w.com", "p@q.com"]})
    report = build_profiling_report(df, with_figures=False)
    for key in ["dataset", "generated_at", "metadata", "profiling", "rules"]:
        assert key in report


def test_report_row_count_matches():
    df = pd.DataFrame({"a": range(42)})
    report = build_profiling_report(df, with_figures=False)
    assert report["dataset"]["rows"] == 42