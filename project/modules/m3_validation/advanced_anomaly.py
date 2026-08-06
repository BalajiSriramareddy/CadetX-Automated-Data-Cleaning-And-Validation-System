"""
MODULE 3 · ANOMALY DETECTION — LOCAL OUTLIER FACTOR & AUTOENCODER
==================================================================
Two further multivariate anomaly methods, alongside Isolation Forest in
anomaly_detection.py. Offered as advanced/optional — see the "which method
when" note at the bottom.

1. LOCAL OUTLIER FACTOR (LOF)
   Measures how isolated a point is relative to its LOCAL neighbourhood density,
   rather than the dataset as a whole. This makes it better than Isolation
   Forest at finding anomalies in data with clusters of different densities —
   e.g. a point that's normal for a low-density region but anomalous for a
   high-density one, which a global method can miss.

2. AUTOENCODER (lightweight, via sklearn's MLPRegressor)
   A neural network trained to reconstruct its own input (input -> compress ->
   reconstruct). It learns the STRUCTURE of normal rows. A row that reconstructs
   poorly (high error) didn't fit the learned pattern — i.e. it's anomalous.
   This uses scikit-learn's MLPRegressor as encoder+decoder in one network,
   rather than TensorFlow/PyTorch, which is heavier than this project needs and
   would add a dependency that's hard to justify for a single component.

WHICH METHOD WHEN (this is the answer to "why three anomaly methods?"):
  - Isolation Forest: fast, general-purpose default. Use first.
  - LOF: when you suspect anomalies are "locally weird" (different density
    regions), not just globally rare.
  - Autoencoder: when relationships between columns are non-linear and complex
    enough that tree-based or distance-based methods miss patterns. Costs more
    to train and is the hardest of the three to explain — use only when the
    simpler methods demonstrably underperform.

Public entry points:
  detect_lof(df, contamination=0.05) -> dict
  detect_autoencoder(df, contamination=0.05) -> dict
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 20
MAX_EXAMPLES = 5


def detect_lof(df: pd.DataFrame, contamination: float = 0.05,
               n_neighbors: int = 20) -> dict:
    """Row-level anomaly detection using Local Outlier Factor.

    contamination: expected proportion of anomalies (an assumption, same
    caveat as Isolation Forest — see anomaly_detection.py).
    n_neighbors: how many nearby points define "local" density.
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] == 0 or len(numeric) < MIN_ROWS:
        return {"method": "local_outlier_factor", "status": "skipped",
                "reason": f"needs >= {MIN_ROWS} rows and numeric columns"}
    if n_neighbors >= len(numeric):
        n_neighbors = max(2, len(numeric) // 2)

    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = model.fit_predict(numeric)     # -1 = anomaly, 1 = normal
    scores = model.negative_outlier_factor_       # more negative = more anomalous

    anomaly_mask = predictions == -1
    anomaly_idx = numeric.index[anomaly_mask].tolist()

    return {
        "method": "local_outlier_factor", "status": "ok",
        "contamination_assumed": contamination, "n_neighbors": n_neighbors,
        "columns_used": list(numeric.columns),
        "rows_scored": int(len(numeric)),
        "anomalies_found": int(anomaly_mask.sum()),
        "anomaly_pct": round(100 * float(anomaly_mask.mean()), 2),
        "example_row_indices": anomaly_idx[:MAX_EXAMPLES],
    }


def detect_autoencoder(df: pd.DataFrame, contamination: float = 0.05,
                       hidden_layer_size: int | None = None,
                       random_state: int = 42) -> dict:
    """Row-level anomaly detection via reconstruction error.

    Trains a small neural network to reconstruct its own (scaled) input through
    a narrower hidden layer — forcing it to learn a compressed representation of
    NORMAL structure. Rows the network reconstructs badly (high error) don't
    fit that structure, so they're flagged as anomalous.

    hidden_layer_size defaults to roughly half the number of input columns
    (the "bottleneck" that forces compression) with a floor of 2.
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2 or len(numeric) < MIN_ROWS:
        return {"method": "autoencoder", "status": "skipped",
                "reason": f"needs >= 2 numeric columns and >= {MIN_ROWS} rows"}

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)

    bottleneck = hidden_layer_size or max(2, numeric.shape[1] // 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPRegressor(
            hidden_layer_sizes=(bottleneck,),   # the "encoding" bottleneck layer
            activation="relu", solver="adam", max_iter=500,
            random_state=random_state, early_stopping=True)
        model.fit(X, X)   # trained to reproduce its OWN input — the autoencoder trick
        reconstructed = model.predict(X)

    reconstruction_error = np.mean((X - reconstructed) ** 2, axis=1)
    threshold = np.quantile(reconstruction_error, 1 - contamination)
    anomaly_mask = reconstruction_error >= threshold
    anomaly_idx = numeric.index[anomaly_mask].tolist()

    return {
        "method": "autoencoder", "status": "ok",
        "contamination_assumed": contamination,
        "bottleneck_size": bottleneck, "columns_used": list(numeric.columns),
        "rows_scored": int(len(numeric)),
        "anomalies_found": int(anomaly_mask.sum()),
        "anomaly_pct": round(100 * float(anomaly_mask.mean()), 2),
        "reconstruction_error_threshold": round(float(threshold), 4),
        "example_row_indices": anomaly_idx[:MAX_EXAMPLES],
        "note": "flags rows the network reconstructs poorly -> don't fit learned structure",
    }