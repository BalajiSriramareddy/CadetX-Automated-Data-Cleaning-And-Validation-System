"""
MODULE 3 · EVALUATION — PRECISION / RECALL / CONFUSION MATRIX / ROC
=====================================================================
Evaluates anomaly detection against KNOWN ground truth.

THE HONEST PROBLEM THIS MODULE FACES: precision, recall, and ROC curves
require knowing which rows are ACTUALLY anomalous — ground truth labels. Real
production data doesn't come with a "this row is wrong" column; if it did, you
wouldn't need anomaly detection in the first place. So this module can only be
run in one of two situations, and it says so honestly rather than faking a
result:

  1. On a SYNTHETIC dataset where anomalies were deliberately planted at known
     row positions (e.g. this project's broadband_customers.csv generator) —
     the planted rows ARE the ground truth.
  2. On a dataset with a genuine labelled subset (e.g. a compliance team has
     manually reviewed and tagged N rows as good/bad).

Without either, this module cannot produce a meaningful score — and it will
say so explicitly rather than silently returning a number that looks
authoritative but means nothing.

Public entry points:
  evaluate_predictions(y_true, y_pred, y_scores=None) -> dict
  evaluate_against_planted_anomalies(df, anomaly_row_indices, predicted_anomaly_indices) -> dict
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)


def evaluate_predictions(y_true: list[int], y_pred: list[int],
                         y_scores: list[float] | None = None) -> dict:
    """Core evaluation given binary ground truth and predictions.

    y_true / y_pred: 1 = anomaly/error, 0 = normal.
    y_scores: continuous anomaly scores (higher = more anomalous), needed for
              the ROC curve; if omitted, ROC is skipped.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2:
        return {"status": "skipped",
                "reason": "ground truth contains only one class — "
                          "precision/recall/ROC are undefined without both classes present"}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "status": "ok",
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "interpretation": {
            "precision": "of the rows flagged as anomalous, what fraction "
                         "actually were — low precision means false alarms",
            "recall": "of the rows that were actually anomalous, what fraction "
                     "were caught — low recall means missed problems",
        },
    }

    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        result["roc"] = {
            "auc": round(float(roc_auc_score(y_true, y_scores)), 4),
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr)//20)]],  # thinned for readability
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr)//20)]],
            "auc_interpretation": (
                "0.5 = no better than random guessing; 1.0 = perfect separation "
                "of anomalous from normal rows"),
        }

    return result


def evaluate_against_planted_anomalies(df_length: int,
                                       true_anomaly_indices: list[int],
                                       predicted_anomaly_indices: list[int],
                                       predicted_scores: dict[int, float] | None = None) -> dict:
    """Convenience wrapper: build y_true/y_pred from two lists of row indices
    (e.g. the rows you deliberately planted as bad vs the rows your anomaly
    detector flagged) and evaluate.

    This is the realistic use case for THIS project: the synthetic dataset
    generator knows exactly which rows it corrupted, so those indices are
    genuine ground truth for a demonstration evaluation.
    """
    y_true = [1 if i in set(true_anomaly_indices) else 0 for i in range(df_length)]
    y_pred = [1 if i in set(predicted_anomaly_indices) else 0 for i in range(df_length)]
    y_scores = None
    if predicted_scores:
        y_scores = [predicted_scores.get(i, 0.0) for i in range(df_length)]
    return evaluate_predictions(y_true, y_pred, y_scores)
