"""
MODULE 2 — TEST SUITE
=====================
Known-answer tests for the cleaning module.

Run:  pytest tests/test_module2.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "modules" / "m2_cleaning"))

from imputation import impute_missing
from deduplication import remove_duplicates
from normalisation import normalise
from quality_score import compute_quality_score, score_before_after
from clean import clean_dataset


# ---------------- imputation ----------------

def test_imputation_fills_numeric_with_median():
    df = pd.DataFrame({"x": [10, 20, None, 40]})  # median of 10,20,40 = 20
    out, _ = impute_missing(df)
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[2] == 20


def test_imputation_fills_categorical_with_mode():
    df = pd.DataFrame({"c": ["a", "a", None, "b"]})
    out, _ = impute_missing(df)
    assert out["c"].iloc[2] == "a"


# ---------------- deduplication ----------------

def test_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    out, _ = remove_duplicates(df)
    assert len(out) == 2


def test_removes_normalised_duplicates():
    df = pd.DataFrame({"name": ["Ava Patel", "  ava patel "]})
    out, _ = remove_duplicates(df)
    assert len(out) == 1


# ---------------- normalisation ----------------

def test_coerces_currency_to_numeric():
    df = pd.DataFrame({"charges": ["£29.99", "£44.99", "54.99"]})
    out, _ = normalise(df)
    assert pd.api.types.is_numeric_dtype(out["charges"])
    assert out["charges"].iloc[0] == 29.99


def test_iso_date_not_corrupted():
    """Critical: 2024-03-10 must stay March, not flip to October."""
    df = pd.DataFrame({"d": ["2024-03-10", "15/02/2024"]})
    out, _ = normalise(df)
    assert out["d"].iloc[0] == "2024-03-10"
    assert out["d"].iloc[1] == "2024-02-15"


def test_unifies_category_case():
    df = pd.DataFrame({"region": ["Oxfordshire", "Oxfordshire", "oxfordshire"]})
    out, _ = normalise(df)
    assert out["region"].nunique() == 1


def test_phone_not_coerced_to_numeric():
    """Identifier guard: a phone column must keep its leading zero, not become a float."""
    df = pd.DataFrame({"phone": ["07123456789", "07999888777", "07111222333"]})
    report = {"metadata": {"phone": {"semantic_type": "phone"}}}
    out, _ = normalise(df, report)
    assert out["phone"].iloc[0] == "07123456789"
    assert not pd.api.types.is_numeric_dtype(out["phone"])


def test_binary_values_unified():
    """Yes/Y/1 -> Yes and No/N/0 -> No."""
    df = pd.DataFrame({"churn": ["Yes", "y", "1", "No", "n", "0"]})
    out, _ = normalise(df)
    assert set(out["churn"].unique()) == {"Yes", "No"}


# ---------------- quality score ----------------

def test_quality_score_in_range():
    df = pd.DataFrame({"a": [1, 2, 3]})
    score = compute_quality_score(df)
    assert 0 <= score["overall_score"] <= 100


def test_cleaning_improves_score():
    before = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", None]})
    after, _ = clean_dataset(before)
    result = score_before_after(before, after)
    assert result["delta"] > 0


# ---------------- integration ----------------

def test_clean_dataset_removes_all_missing_and_dupes():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "charges": ["£10", "£20", "£20", None],
        "region": ["Kent", "kent", "kent", "Devon"],
    })
    out, log = clean_dataset(df)
    assert out.isna().sum().sum() == 0
    assert out.duplicated().sum() == 0
    assert "quality" in log and "steps" in log