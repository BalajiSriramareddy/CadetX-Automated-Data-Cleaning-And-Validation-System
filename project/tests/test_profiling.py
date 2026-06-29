"""Starter test for Module 1. Expand in Week 5.

Run with:  pytest
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "modules"))
from m1_profiling.profile import missing_values, basic_shape  # noqa: E402


def test_basic_shape_counts_rows_and_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    shape = basic_shape(df)
    assert shape["rows"] == 3
    assert shape["columns"] == 2


def test_missing_values_detects_gaps():
    df = pd.DataFrame({"a": [1, None, 3]})
    result = missing_values(df)
    assert result["a"]["missing"] == 1
