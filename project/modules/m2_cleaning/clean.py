"""
MODULE 2 · FILE 5 — CLEANING API (ORCHESTRATOR)
===============================================
Chains cleaning steps and writes cleaned_data.csv + cleaning_log.json.

CORE pipeline (always runs):
  1. deduplicate (exact + normalised)
  2. normalise (types/dates/categories)
  3. impute (statistical: median/mode)
  4. score (before/after quality)

OPTIONAL advanced steps (config-controlled, OFF by default):
  - fuzzy duplicate matching   (flags candidates only, never auto-merges)
  - ML-based imputation        (replaces statistical imputation when enabled)
  - schema inference           (attached to the cleaning log for reference)
  - transformation engine      (scaling/encoding/features — for ML use, not
                                 needed for a human-readable analyst report)

Defaults are conservative and explainable; advanced options are there for a
justified, specific need — see each module's own docstring for why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))

from deduplication import remove_duplicates
from normalisation import normalise
from imputation import impute_missing
from quality_score import score_before_after

# optional/advanced components
from fuzzy_matching import find_fuzzy_duplicates
from ml_imputation import impute_ml
from schema_inference import infer_schema
from transformation import transform


def clean_dataset(df: pd.DataFrame, report: dict | None = None,
                  options: dict | None = None):
    """Run the cleaning pipeline. `options` (all default False/off):
      fuzzy_duplicate_check : bool  -> also run fuzzy matching (flag-only)
      ml_imputation         : str|None -> "knn"|"regression"|"iterative" to
                                          REPLACE statistical imputation
      infer_schema          : bool  -> attach an inferred schema to the log
      transform              : bool  -> run scaling/encoding/feature extraction
                                        on the OUTPUT (adds columns, does not
                                        replace the cleaned human-readable data)
    Returns (cleaned_df, log_dict).
    """
    options = options or {}
    before = df.copy()

    df, dedup_log = remove_duplicates(df, report)
    df, norm_log = normalise(df, report)

    if options.get("ml_imputation"):
        df, impute_log = impute_ml(df, method=options["ml_imputation"])
    else:
        df, impute_log = impute_missing(df, report)

    quality = score_before_after(before, df)

    from datetime import datetime
    log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_in": int(len(before)),
        "rows_out": int(len(df)),
        "quality": quality,
        "steps": {
            "deduplication": dedup_log,
            "normalisation": norm_log,
            "imputation": impute_log,
        },
        "advanced": {},
    }

    if options.get("fuzzy_duplicate_check"):
        log["advanced"]["fuzzy_duplicates"] = find_fuzzy_duplicates(df, report=report)

    if options.get("infer_schema"):
        log["advanced"]["inferred_schema"] = infer_schema(df, report)

    if options.get("transform"):
        df, transform_log = transform(df, report)
        log["advanced"]["transformation"] = transform_log

    return df, log