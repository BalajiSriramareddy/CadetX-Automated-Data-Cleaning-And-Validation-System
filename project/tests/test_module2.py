"""
MODULE 2 — TEST SUITE
----------------------
Known-answer tests for the cleaning module.

Run:  pytest tests/test_module2.py -v
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "modules" / "m2_cleaning"))

from clean import (quality_score, drop_duplicates, normalise, impute_missing,
                    _fix_numeric_as_text, _fix_binary_categorical,
                    _fix_dates, UK_PHONE_RE)


def test_perfect_dataset_scores_100():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert quality_score(df) == 100.0


def test_missing_and_duplicates_lower_the_score():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "z"]})
    assert quality_score(df) < 100.0


def test_exact_duplicates_removed():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, log = drop_duplicates(df)
    assert len(cleaned) == 2
    assert log["exact_duplicates_removed"] == 1


def test_currency_symbols_stripped_to_float():
    s = pd.Series(["£29.99", "44.99", "£54.99"])
    out = _fix_numeric_as_text(s)
    assert out.tolist() == [29.99, 44.99, 54.99]


def test_normalise_does_not_touch_phone_even_if_numeric_as_text():
    """Regression test: a phone column can ALSO get flagged
    'numeric_as_text' by Module 1, but must never be coerced to float.
    """
    df = pd.DataFrame({"phone": ["07123456789", "07xx", "07234567890"]})
    report = {
        "metadata": {
            "phone": {"inferred_type": "numeric_as_text", "semantic_type": "phone"}
        },
        "rules": {"inconsistent_formats": {}},
    }
    cleaned, changes = normalise(df, report=report)
    assert not pd.api.types.is_float_dtype(cleaned["phone"])
    assert (cleaned["phone"] == "INVALID_PHONE").sum() == 1
    assert changes["phone"]["count"] == 1


def test_yes_no_style_spellings_unified():
    s = pd.Series(["Yes", "yes", "Y", "1", "No", "no", "N", "0"])
    out = _fix_binary_categorical(s)
    assert out.tolist() == ["Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No"]


def test_impute_missing_fills_all_nulls():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "x"]})
    cleaned, log = impute_missing(df)
    assert cleaned.isna().sum().sum() == 0
    assert len(log) == 2
def test_ambiguous_uk_date_parsed_dayfirst():
    """'03/04/2023' must parse as 3 April, not March 4th."""
    s = pd.Series(["03/04/2023"])
    result = _fix_dates(s)
    assert result.iloc[0].day == 3
    assert result.iloc[0].month == 4


def test_unambiguous_date_still_correct():
    """Day > 12 leaves no ambiguity — sanity check dayfirst didn't break this."""
    s = pd.Series(["25/12/2023"])
    result = _fix_dates(s)
    assert result.iloc[0].day == 25
    assert result.iloc[0].month == 12


def test_uk_phone_regex_rejects_nine_digits():
    """A number one digit short of a valid UK mobile should not match."""
    assert UK_PHONE_RE.match("0712345678") is None  # 9 digits after 0


def test_uk_phone_regex_accepts_ten_digits():
    assert UK_PHONE_RE.match("07123456789") is not None  # 10 digits after 0  
  
  
  
  
  
