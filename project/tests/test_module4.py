"""
MODULE 4 — INTEGRATION TEST SUITE
=================================
Unit tests check a component in isolation. INTEGRATION tests check the seams
BETWEEN components — which is where this project's worst bugs were found
(read-time type coercion, and one bad value destroying a whole column).

Run:  pytest tests/test_module4.py -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "modules" / "m4_pipeline", ROOT / "modules"):
    sys.path.append(str(p))

from pipeline import run_pipeline, load_config, _deep_merge          # noqa: E402
from schemas import (validate_input, validate_stage_output,          # noqa: E402
                     SchemaError, describe_schemas)
from logger import get_logger, log_data_quality_warnings             # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def messy_csv(tmp_path):
    """A small dataset with known planted problems."""
    df = pd.DataFrame({
        "customer_id": [f"1000{i}" for i in range(30)],
        "email": (["a@x.com"] * 25) + ["bad-email"] * 5,
        "phone": ["07123456789"] * 30,                 # leading zero must survive
        "postcode": (["OX7 3AB"] * 28) + ["12345", "OX1 2JD"],
        "nps_score": [str(i % 11) for i in range(30)],
        "churn": (["Yes"] * 15) + (["no"] * 10) + (["Y"] * 5),
        "monthly_charges": (["29.99"] * 20) + (["£44.99"] * 10),
    })
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)   # 3 duplicate rows
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def run_config(tmp_path):
    return {
        "paths": {"output_dir": str(tmp_path / "outputs"),
                  "processed_dir": str(tmp_path / "processed"),
                  "log_dir": str(tmp_path / "logs"), "rules_file": None},
        "profiling": {"generate_figures": False, "figures_subdir": "figures"},
        "validation": {"contamination": 0.05, "apply_declared_rules": False},
        "thresholds": {"fail_on_verdict_fail": True},
        "logging": {"verbose": False},
    }


# --------------------------------------------------------------------------- #
# end-to-end pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_runs_end_to_end(messy_csv, run_config):
    summary = run_pipeline(messy_csv, run_config)
    assert summary["verdict"] in ("PASS", "WARNING", "FAIL")
    assert summary["rows_in"] == 33
    assert summary["rows_out"] == 30          # 3 duplicates removed


def test_pipeline_writes_all_four_artefacts(messy_csv, run_config):
    summary = run_pipeline(messy_csv, run_config)
    for name, path in summary["outputs"].items():
        assert Path(path).exists(), f"{name} was not written"


def test_pipeline_improves_quality(messy_csv, run_config):
    summary = run_pipeline(messy_csv, run_config)
    assert summary["quality_after"] > summary["quality_before"]


def test_leading_zeros_survive_the_whole_pipeline(messy_csv, run_config):
    """Regression: pandas read_csv used to coerce phones to int and drop the 0."""
    summary = run_pipeline(messy_csv, run_config)
    cleaned = pd.read_csv(summary["outputs"]["cleaned_data"], dtype=str)
    assert cleaned["phone"].iloc[0] == "07123456789"


def test_text_column_not_destroyed_by_one_numeric_value(messy_csv, run_config):
    """Regression: a postcode column containing '12345' was being coerced to
    numeric, turning every real postcode into NaN."""
    summary = run_pipeline(messy_csv, run_config)
    cleaned = pd.read_csv(summary["outputs"]["cleaned_data"], dtype=str)
    assert "OX7 3AB" in set(cleaned["postcode"].dropna())


def test_stage_outputs_are_valid_json(messy_csv, run_config):
    summary = run_pipeline(messy_csv, run_config)
    for key in ("profiling_report", "cleaning_log", "validation_report"):
        with open(summary["outputs"][key], encoding="utf-8") as f:
            assert isinstance(json.load(f), dict)


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def test_input_schema_rejects_empty_dataframe():
    with pytest.raises(SchemaError):
        validate_input(pd.DataFrame())


def test_input_schema_rejects_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(SchemaError):
        validate_input(df)


def test_stage_schema_rejects_missing_keys():
    with pytest.raises(SchemaError):
        validate_stage_output("profiling", {"dataset": {}})


def test_stage_schema_accepts_valid_structure():
    validate_stage_output("summary", {
        "input": "x.csv", "rows_in": 1, "rows_out": 1,
        "verdict": "PASS", "outputs": {}})


def test_describe_schemas_documents_all_stages():
    described = describe_schemas()
    assert set(described["stages"]) == {"profiling", "cleaning",
                                        "validation", "summary"}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_config_falls_back_to_defaults_when_missing():
    config = load_config("does/not/exist.yaml")
    assert config["validation"]["contamination"] == 0.05


def test_deep_merge_overrides_nested_values():
    merged = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 99}})
    assert merged["a"] == {"x": 1, "y": 99}


def test_config_file_is_readable(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("validation:\n  contamination: 0.2\n")
    assert load_config(cfg)["validation"]["contamination"] == 0.2


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def test_logger_writes_a_log_file(tmp_path):
    log = get_logger("test", tmp_path / "logs", verbose=False)
    log.info("hello")
    assert list((tmp_path / "logs").glob("*.log"))


def test_quality_warnings_raised_from_findings(tmp_path):
    log = get_logger("test2", tmp_path / "logs", verbose=False)
    report = {
        "profiling": {"missing_values": {"per_column": {"x": {"missing_pct": 50.0}}}},
        "rules": {"inconsistent_formats": {}, "potential_pii": {}},
    }
    assert log_data_quality_warnings(log, report) >= 1