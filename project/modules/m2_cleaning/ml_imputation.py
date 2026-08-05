"""
MODULE 2 · ML-BASED IMPUTATION
==============================
Three ML imputation methods, offered as an UPGRADE PATH alongside the default
statistical imputation (median/mode) in imputation.py.

  1. KNN imputer          - fills a gap using the average of the k most similar
                            rows (by their other numeric columns).
  2. Regression imputer   - fits a linear model predicting the missing column
                            from the other numeric columns, then predicts the gaps.
  3. Iterative imputer    - repeats the regression idea round-robin across every
                            column with gaps, each round using the others'
                            current best-guess values ("MICE"-style).

WHY THESE ARE OPTIONAL, NOT THE DEFAULT (documented, not hidden):
  - They only use NUMERIC columns and require at least 2 numeric columns with
    some overlap in what's missing — many real datasets don't have that shape.
  - They are much harder to explain to a stakeholder than "we used the median."
    If asked "why is this customer's NPS 7.4?", "the model predicted it from
    their other numeric fields" is a weaker, less auditable answer than
    "it's the column median."
  - They can quietly encode bias: if the model learns that a certain postcode
    correlates with lower charges, it will impute lower charges for any missing
    value in that postcode — reproducing a pattern that might not be causal.
  - They are slower and require scikit-learn's experimental IterativeImputer.

Median/mode remains the DEFAULT in the pipeline. These are available for a
data scientist who has a specific, justified reason to want them, and who can
explain the trade-off above.

Public entry point:  impute_ml(df, method="knn", columns=None) -> (df, log)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required side-effect import)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression

MIN_NUMERIC_COLUMNS = 2   # ML imputers need other columns to learn a relationship from
DEFAULT_KNN_NEIGHBOURS = 5


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns only — ML imputers here operate on numeric data.
    (Categorical ML imputation exists but adds real complexity for little gain
    over mode-fill; deliberately out of scope — see imputation.py's docstring.)"""
    return df.select_dtypes(include="number")


def _eligible(df: pd.DataFrame) -> tuple[bool, str]:
    num = _numeric_frame(df)
    if num.shape[1] < MIN_NUMERIC_COLUMNS:
        return False, (f"needs >= {MIN_NUMERIC_COLUMNS} numeric columns, "
                       f"found {num.shape[1]}")
    if num.isna().sum().sum() == 0:
        return False, "no missing numeric values to impute"
    return True, "ok"


def impute_knn(df: pd.DataFrame, n_neighbors: int = DEFAULT_KNN_NEIGHBOURS) -> tuple[pd.DataFrame, dict]:
    """Fill numeric gaps using the average of the k most similar rows.

    'Similar' means closest in the other numeric columns (Euclidean distance).
    Sensible when rows with similar profiles tend to have similar values for
    the missing field (e.g. similar customers have similar spend).
    """
    df = df.copy()
    ok, reason = _eligible(df)
    if not ok:
        return df, {"method": "knn", "status": "skipped", "reason": reason}

    num = _numeric_frame(df)
    before_missing = num.isna().sum().sum()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled = imputer.fit_transform(num)
    df[num.columns] = filled

    return df, {"method": "knn", "status": "ok", "n_neighbors": n_neighbors,
                "values_filled": int(before_missing),
                "columns_used": list(num.columns)}


def impute_regression(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    """Fill one column's gaps by predicting it from the other numeric columns.

    Fits a linear regression on rows where target_col is present, using the
    other numeric columns as predictors, then predicts the missing rows.
    """
    df = df.copy()
    num = _numeric_frame(df)
    if target_col not in num.columns:
        return df, {"method": "regression", "status": "skipped",
                    "reason": f"'{target_col}' is not numeric"}
    predictors = [c for c in num.columns if c != target_col]
    if len(predictors) == 0:
        return df, {"method": "regression", "status": "skipped",
                    "reason": "no other numeric columns to predict from"}

    missing_mask = df[target_col].isna()
    if missing_mask.sum() == 0:
        return df, {"method": "regression", "status": "skipped",
                    "reason": f"'{target_col}' has no missing values"}

    train = num.dropna(subset=[target_col] + predictors)
    if len(train) < 10:
        return df, {"method": "regression", "status": "skipped",
                    "reason": "fewer than 10 complete rows to train on"}

    model = LinearRegression()
    model.fit(train[predictors], train[target_col])

    predict_rows = df.loc[missing_mask, predictors].dropna()
    if predict_rows.empty:
        return df, {"method": "regression", "status": "skipped",
                    "reason": "rows with missing target also missing predictors"}

    predictions = model.predict(predict_rows)
    df.loc[predict_rows.index, target_col] = predictions

    return df, {"method": "regression", "status": "ok", "target": target_col,
                "predictors": predictors, "values_filled": int(len(predictions)),
                "r_squared_on_training_data": round(float(model.score(
                    train[predictors], train[target_col])), 3)}


def impute_iterative(df: pd.DataFrame, max_iter: int = 10) -> tuple[pd.DataFrame, dict]:
    """Round-robin imputation: each numeric column with gaps is predicted from
    all others in turn, repeating until predictions stabilise (or max_iter).

    This is the most sophisticated of the three, and the hardest to explain —
    use only when a single KNN/regression pass isn't good enough and you can
    justify the added complexity.
    """
    df = df.copy()
    ok, reason = _eligible(df)
    if not ok:
        return df, {"method": "iterative", "status": "skipped", "reason": reason}

    num = _numeric_frame(df)
    before_missing = int(num.isna().sum().sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,
                                   random_state=42)
        filled = imputer.fit_transform(num)
    df[num.columns] = filled

    return df, {"method": "iterative", "status": "ok", "max_iter": max_iter,
                "values_filled": before_missing, "columns_used": list(num.columns),
                "converged": bool(imputer.n_iter_ < max_iter)}


def impute_ml(df: pd.DataFrame, method: str = "knn",
             columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Dispatch to the chosen ML imputation method.

    method: "knn" | "regression" | "iterative"
    columns: for "regression" only — which column(s) to predict; if None, runs
             regression for every numeric column that has missing values.
    """
    if method == "knn":
        return impute_knn(df)
    if method == "iterative":
        return impute_iterative(df)
    if method == "regression":
        num = _numeric_frame(df)
        targets = columns or [c for c in num.columns if df[c].isna().any()]
        logs = []
        for col in targets:
            df, log = impute_regression(df, col)
            logs.append(log)
        return df, {"method": "regression", "per_column": logs}
    raise ValueError(f"unknown method '{method}'; choose knn, regression, or iterative")