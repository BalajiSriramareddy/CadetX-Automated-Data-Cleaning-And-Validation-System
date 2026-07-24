"""
MODULE 4 · LOGGING SYSTEM
=========================
Structured INFO / WARNING / ERROR logging for the pipeline.

WHY not print()? Because in production you need to know WHEN something happened,
HOW serious it was, and you need a durable record. print() gives none of that.
This module writes to two places at once:

  console  -> readable progress for a human running the pipeline
  file     -> logs/pipeline_<timestamp>.log, a permanent audit trail

LEVELS AS USED HERE
  INFO     normal progress            "cleaning complete, 420 -> 400 rows"
  WARNING  suspicious but not fatal    "18% of emails are invalid"
  ERROR    the stage failed            "input file has no columns"

Public entry points:
  get_logger(name, log_dir, verbose) -> configured Logger
  log_data_quality_warnings(logger, report) -> raises WARNINGs from findings
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

# thresholds at which a finding is worth a WARNING rather than just INFO
WARN_MISSING_PCT = 20.0
WARN_INVALID_PCT = 10.0


def get_logger(name: str = "pipeline",
               log_dir: str | Path = "logs",
               verbose: bool = True) -> logging.Logger:
    """Return a logger that writes to the console and to a timestamped file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()          # avoid duplicate handlers on repeat calls
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"pipeline_{stamp}.log",
                                       encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)     # file keeps everything
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("logger initialised -> %s", log_path / f"pipeline_{stamp}.log")
    return logger


def log_data_quality_warnings(logger: logging.Logger, profiling_report: dict) -> int:
    """Turn profiling findings into WARNING lines. Returns the warning count.

    This is what makes the log useful rather than decorative: real problems in
    the data surface as warnings a human can act on.
    """
    count = 0
    missing = profiling_report.get("profiling", {}).get("missing_values", {})
    for col, stats in missing.get("per_column", {}).items():
        if stats.get("missing_pct", 0) >= WARN_MISSING_PCT:
            logger.warning("column '%s' is %.1f%% missing", col, stats["missing_pct"])
            count += 1

    formats = profiling_report.get("rules", {}).get("inconsistent_formats", {})
    for col, detail in formats.items():
        pct = detail.get("invalid_pct")
        if pct is not None and pct >= WARN_INVALID_PCT:
            logger.warning("column '%s': %.1f%% of values fail %s format",
                           col, pct, detail.get("expected_format", "expected"))
            count += 1
        elif detail.get("issue"):
            logger.warning("column '%s': %s", col, detail["issue"])
            count += 1

    pii = profiling_report.get("rules", {}).get("potential_pii", {})
    if pii:
        logger.warning("potential personal data in %d column(s): %s",
                       len(pii), ", ".join(pii.keys()))
        count += 1

    return count