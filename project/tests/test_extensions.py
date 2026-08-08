"""
MODULE 2 & 3 EXTENSIONS — TEST SUITE
====================================
Covers every component added to reach full completeness against the brief:
schema inference, fuzzy matching, ML imputation, transformation (M2), and
LOF/autoencoder, NLP classifier, AI error detection, evaluation (M3).

Run:  pytest tests/test_extensions.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "modules" / "m2_cleaning"))
sys.path.append(str(ROOT / "modules" / "m3_validation"))

from schema_inference import infer_schema
from fuzzy_matching import find_fuzzy_duplicates
from ml_imputation import impute_knn, impute_regression, impute_iterative, impute_ml
from transformation import scale_numeric, encode_categorical, extract_date_features, extract_ratio_feature, transform
from advanced_anomaly import detect_lof, detect_autoencoder
from nlp_classifier import classify_column_meaning, detect_mislabeled_columns, classify_columns
from ai_error_detection import detect_impossible_values, detect_suspicious_patterns, detect_data_drift
from evaluation import evaluate_predictions, evaluate_against_planted_anomalies


# ---------------- schema_inference ----------------

def test_infers_numeric_range():
    df = pd.DataFrame({"nps": ["8", "9", "10", "7"]})
    schema = infer_schema(df)
    assert schema["columns"]["nps"]["expected_range"] == {"min": 7.0, "max": 10.0}


def test_infers_email_format():
    df = pd.DataFrame({"email": ["a@x.com", "b@y.com"]})
    report = {"metadata": {"email": {"semantic_type": "email"}}}
    schema = infer_schema(df, report)
    assert schema["columns"]["email"]["expected_format"] == "email"


def test_infers_allowed_values_for_category():
    df = pd.DataFrame({"churn": ["Yes", "No", "Yes", "No"]})
    schema = infer_schema(df)
    assert set(schema["columns"]["churn"]["allowed_values"]) == {"Yes", "No"}


# ---------------- fuzzy_matching ----------------

def test_finds_name_typo():
    df = pd.DataFrame({"full_name": ["John Smith", "Jon Smith", "Ava Patel"]})
    report = {"metadata": {"full_name": {"semantic_type": "person_name"}}}
    result = find_fuzzy_duplicates(df, report=report)
    assert result["total_candidate_pairs"] >= 1


def test_never_fuzzy_matches_emails():
    df = pd.DataFrame({"email": ["a@x.com", "a@x.co"]})  # similar but must not be checked
    report = {"metadata": {"email": {"semantic_type": "email"}}}
    result = find_fuzzy_duplicates(df, report=report)
    assert "email" not in result["columns_checked"]


def test_exact_matches_not_reported_as_fuzzy():
    """Case-only differences are exact-dedup's job, not fuzzy matching's."""
    df = pd.DataFrame({"full_name": ["Ava Patel", "ava patel"]})
    report = {"metadata": {"full_name": {"semantic_type": "person_name"}}}
    result = find_fuzzy_duplicates(df, report=report)
    assert result["total_candidate_pairs"] == 0


# ---------------- ml_imputation ----------------

def test_knn_fills_all_gaps():
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    out, log = impute_knn(df)
    assert out.isna().sum().sum() == 0
    assert log["status"] == "ok"


def test_knn_skips_single_numeric_column():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out, log = impute_knn(df)
    assert log["status"] == "skipped"


def test_regression_reports_fit_quality():
    x = np.arange(50, dtype=float)
    y = x * 2 + 1
    df = pd.DataFrame({"x": x, "y": y})
    df.loc[5:9, "y"] = None
    out, log = impute_regression(df, "y")
    assert log["status"] == "ok"
    assert log["r_squared_on_training_data"] > 0.9


def test_iterative_fills_gaps():
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(0, 1, 30),
                       "b": np.random.default_rng(1).normal(0, 1, 30)})
    df.loc[0:4, "a"] = None
    out, log = impute_ml(df, method="iterative")
    assert out["a"].isna().sum() == 0


# ---------------- transformation ----------------

def test_scaling_produces_mean_zero():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    out, log = scale_numeric(df)
    assert abs(out["x_scaled"].mean()) < 1e-9


def test_original_columns_preserved_after_scaling():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out, _ = scale_numeric(df)
    assert "x" in out.columns and "x_scaled" in out.columns


def test_one_hot_encoding_creates_dummy_columns():
    df = pd.DataFrame({"region": ["A", "B", "A", "C"]})
    out, log = encode_categorical(df)
    assert "region_A" in out.columns and "region_B" in out.columns


def test_dates_not_one_hot_encoded():
    """Regression: dates were briefly being one-hot encoded as if categorical."""
    df = pd.DataFrame({"install_date": ["2024-01-15", "2024-02-20", "2024-03-10"]})
    report = {"metadata": {"install_date": {"semantic_type": "date"}}}
    out, log = encode_categorical(df, report=report)
    assert not any(c.startswith("install_date_2024") for c in out.columns)


def test_date_feature_extraction():
    df = pd.DataFrame({"install_date": ["2024-01-15", "2024-06-20"]})
    out, log = extract_date_features(df)
    assert "install_date_month" in out.columns
    assert out["install_date_month"].iloc[0] == 1


def test_ratio_feature_handles_divide_by_zero():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 5.0]})
    out, _ = extract_ratio_feature(df, "a", "b")
    assert pd.isna(out["a_per_b"].iloc[0])
    assert out["a_per_b"].iloc[1] == 4.0


# ---------------- advanced_anomaly ----------------

def test_lof_detects_outlier():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
    df.loc[0, "x"] = 50   # extreme outlier
    result = detect_lof(df)
    assert result["status"] == "ok"
    assert 0 in result["example_row_indices"]


def test_autoencoder_runs_and_flags_outlier():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 100), "y": rng.normal(0, 1, 100)})
    df.loc[0, "x"] = 50
    result = detect_autoencoder(df)
    assert result["status"] == "ok"
    assert result["anomalies_found"] > 0


def test_lof_skips_tiny_data():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert detect_lof(df)["status"] == "skipped"


# ---------------- nlp_classifier ----------------

def test_classifies_email_column_name():
    m = classify_column_meaning("customer_email")
    assert m["predicted_meaning"] == "email"


def test_classifies_camelcase_name():
    m = classify_column_meaning("monthlyCharges")
    assert m["predicted_meaning"] == "money"


def test_detects_mislabeled_column():
    df = pd.DataFrame({"customer_email": ["Yes", "No", "Yes", "No", "Yes"]})
    findings = detect_mislabeled_columns(df)
    assert "customer_email" in findings


def test_correctly_labelled_column_not_flagged():
    df = pd.DataFrame({"email": [f"user{i}@x.com" for i in range(10)]})
    findings = detect_mislabeled_columns(df)
    assert "email" not in findings


# ---------------- ai_error_detection ----------------

def test_detects_negative_age():
    df = pd.DataFrame({"customer_age": [25, -5, 40]})
    findings = detect_impossible_values(df)
    assert "customer_age" in findings


def test_detects_future_date():
    df = pd.DataFrame({"install_date": ["2024-01-15", "2099-01-01"]})
    findings = detect_impossible_values(df)
    assert "install_date" in findings


def test_detects_round_number_fingerprint():
    df = pd.DataFrame({"suspicious": [100, 200, 300, 400, 500] * 5})
    findings = detect_suspicious_patterns(df)
    assert "suspicious" in findings


def test_drift_detected_on_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"charges": rng.normal(45, 8, 200)})
    cur = pd.DataFrame({"charges": rng.normal(80, 8, 200)})
    result = detect_data_drift(ref, cur)
    assert "charges" in result["columns_with_drift"]


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"charges": rng.normal(45, 8, 200)})
    cur = pd.DataFrame({"charges": rng.normal(45, 8, 200)})
    result = detect_data_drift(ref, cur)
    assert "charges" not in result["columns_with_drift"]


# ---------------- evaluation ----------------

def test_confusion_matrix_correct():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    result = evaluate_predictions(y_true, y_pred)
    assert result["confusion_matrix"]["true_positive"] == 1
    assert result["confusion_matrix"]["false_negative"] == 1


def test_single_class_ground_truth_is_refused():
    result = evaluate_predictions([0, 0, 0], [0, 1, 0])
    assert result["status"] == "skipped"


def test_roc_auc_perfect_separation():
    y_true = [0]*5 + [1]*5
    y_pred = [0]*5 + [1]*5
    y_scores = [0.1]*5 + [0.9]*5
    result = evaluate_predictions(y_true, y_pred, y_scores)
    assert result["roc"]["auc"] == 1.0


def test_wrapper_builds_correct_labels():
    result = evaluate_against_planted_anomalies(
        df_length=10, true_anomaly_indices=[0, 1], predicted_anomaly_indices=[0, 1, 5])
    assert result["confusion_matrix"]["true_positive"] == 2
    assert result["confusion_matrix"]["false_positive"] == 1
