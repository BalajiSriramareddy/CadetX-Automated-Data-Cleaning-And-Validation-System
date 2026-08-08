"""
MODULE 4 — INTEGRATION, ORCHESTRATION & PRODUCTION ENGINEERING
================================================================
ONE configurable pipeline: profiling -> cleaning -> validation, single command.

  raw.csv
     |
     v  [M1 Profile ] -> outputs/profiling_report.json  (+ figures/)
     v  [M2 Clean   ] -> data/processed/cleaned_data.csv + cleaning_log.json
     v  [M3 Validate] -> outputs/validation_report.json
     v
  pipeline_summary.json  -> PASS / WARNING / FAIL

PRODUCTION FEATURES
  configuration : pipeline_config.yaml, CLI flags override the file
  logging       : INFO/WARNING/ERROR, console + logs/pipeline_<ts>.log
  schemas       : every stage's input/output is contract-checked
  exit codes    : 0 = PASS/WARNING, 1 = FAIL, 2 = pipeline error

CLI
    python pipeline.py --input data.csv --output results/
    python pipeline.py --config modules/m4_pipeline/pipeline_config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                         # project/
MODULES = ROOT / "modules"

for p in (HERE, MODULES, MODULES / "m1_profiling",
          MODULES / "m2_cleaning", MODULES / "m3_validation"):
    sys.path.append(str(p))

from profiling_api import build_profiling_report     # noqa: E402  (M1)
from clean import clean_dataset                        # noqa: E402  (M2)
from validate import build_validation_report            # noqa: E402  (M3)
from schemas import validate_input, validate_stage_output, SchemaError  # noqa: E402
from logger import get_logger, log_data_quality_warnings                # noqa: E402

DEFAULT_CONFIG = {
    "paths": {"output_dir": "outputs", "processed_dir": "data/processed",
              "log_dir": "logs", "rules_file": None, "input": None},
    "profiling": {"generate_figures": True, "figures_subdir": "figures"},
    "cleaning": {"fuzzy_duplicate_check": False, "ml_imputation": None,
                "infer_schema": True, "transform": False},
    "validation": {"contamination": 0.05, "apply_declared_rules": True,
                   "local_outlier_factor": False, "autoencoder": False,
                   "nlp_classifier": False, "ai_error_detection": False,
                   "data_drift": False, "evaluate": False},
    "thresholds": {"pass_score": 90, "warning_score": 70,
                   "fail_on_verdict_fail": True},
    "logging": {"verbose": True},
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str | Path | None = None) -> dict:
    """Built-in defaults <- pipeline_config.yaml <- CLI flags (later wins)."""
    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    if not config_path:
        return config
    path = Path(config_path)
    if not path.exists():
        return config
    if yaml is None:
        raise ImportError("pyyaml is required to read a config file")
    with open(path, encoding="utf-8") as f:
        return _deep_merge(config, yaml.safe_load(f) or {})


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def run_pipeline(input_path: str | Path, config: dict | None = None) -> dict:
    """Run profiling -> cleaning -> validation. Returns the summary dict."""
    cfg = config or {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    paths = cfg["paths"]

    out_dir = Path(paths.get("output_dir", "outputs"))
    proc_dir = Path(paths.get("processed_dir", "data/processed"))
    log_dir = Path(paths.get("log_dir", "logs"))
    rules_file = paths.get("rules_file") if cfg["validation"].get(
        "apply_declared_rules", True) else None

    log = get_logger("pipeline", log_dir, cfg.get("logging", {}).get("verbose", True))
    started = time.time()
    input_path = Path(input_path)

    log.info("=" * 62)
    log.info("AUTOMATED DATA QUALITY PIPELINE")
    log.info("input: %s", input_path)
    log.info("=" * 62)

    # dtype=str deliberately: pandas' default inference reads "07123456789" as
    # an integer and drops the leading zero. Types are inferred later, visibly.
    df = pd.read_csv(input_path, dtype=str)
    validate_input(df, source=input_path.name)
    log.info("loaded %d rows x %d columns", len(df), df.shape[1])

    # ---------------- MODULE 1: PROFILE ----------------
    log.info("[1/3] Profiling...")
    profiling = build_profiling_report(
        df, source=str(input_path),
        with_figures=cfg["profiling"].get("generate_figures", True),
        figures_dir=out_dir / cfg["profiling"].get("figures_subdir", "figures"))
    validate_stage_output("profiling", profiling)
    _write_json(out_dir / "profiling_report.json", profiling)

    pii = list(profiling["rules"]["potential_pii"].keys())
    suspicious = list(profiling["rules"]["suspicious_columns"].keys())
    log.info("      PII columns flagged: %s", pii or "none")
    if suspicious:
        log.warning("      suspicious columns: %s", suspicious)
    n_warnings = log_data_quality_warnings(log, profiling)
    log.info("      profiling complete (%d data-quality warnings raised)", n_warnings)

    # ---------------- MODULE 2: CLEAN ----------------
    log.info("[2/3] Cleaning...")
    cleaning_options = {
        k: v for k, v in cfg.get("cleaning", {}).items()
        if k in ("fuzzy_duplicate_check", "ml_imputation", "infer_schema", "transform")
    }
    cleaned, cleaning_log = clean_dataset(df, profiling, options=cleaning_options)
    validate_stage_output("cleaning", cleaning_log)
    proc_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = proc_dir / "cleaned_data.csv"
    cleaned.to_csv(cleaned_path, index=False)
    _write_json(out_dir / "cleaning_log.json", cleaning_log)
    q = cleaning_log["quality"]
    log.info("      rows %d -> %d | quality %.2f -> %.2f (+%.2f)",
             cleaning_log["rows_in"], cleaning_log["rows_out"],
             q["score_before"], q["score_after"], q["delta"])

    # ---------------- MODULE 3: VALIDATE ----------------
    log.info("[3/3] Validating...")
    validation_options = {
        k: v for k, v in cfg.get("validation", {}).items()
        if k in ("local_outlier_factor", "autoencoder", "nlp_classifier",
                 "ai_error_detection", "data_drift", "evaluate")
    }
    validation = build_validation_report(
        cleaned, profiling, rules_file,
        cfg["validation"].get("contamination", 0.05), source=str(cleaned_path),
        options=validation_options)
    validate_stage_output("validation", validation)
    _write_json(out_dir / "validation_report.json", validation)

    health = validation["health"]
    rv = validation["rule_validation"]["summary"]
    if rv["total_violations"]:
        log.warning("      %d rule violations across %d columns",
                    rv["total_violations"], len(rv["columns_with_violations"]))
    log.info("      health score %.2f -> %s", health["health_score"], health["verdict"])

    # ---------------- SUMMARY ----------------
    elapsed = round(time.time() - started, 2)
    summary = {
        "input": str(input_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_in": int(len(df)),
        "rows_out": int(len(cleaned)),
        "quality_before": q["score_before"],
        "quality_after": q["score_after"],
        "health_score": health["health_score"],
        "verdict": health["verdict"],
        "rule_violations": rv["total_violations"],
        "warnings_raised": n_warnings,
        "elapsed_seconds": elapsed,
        "outputs": {
            "profiling_report": str(out_dir / "profiling_report.json"),
            "cleaned_data": str(cleaned_path),
            "cleaning_log": str(out_dir / "cleaning_log.json"),
            "validation_report": str(out_dir / "validation_report.json"),
        },
    }
    validate_stage_output("summary", summary)
    _write_json(out_dir / "pipeline_summary.json", summary)

    if health["verdict"] == "FAIL":
        log.error("VERDICT: FAIL - %s", health["reason"])
    elif health["verdict"] == "WARNING":
        log.warning("VERDICT: WARNING - %s", health["reason"])
    else:
        log.info("VERDICT: PASS - %s", health["reason"])
    log.info("pipeline complete in %.2fs", elapsed)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Automated Data Quality Pipeline (profile -> clean -> validate)")
    ap.add_argument("--input", help="Path to the raw CSV")
    ap.add_argument("--output", "--output-dir", dest="output_dir",
                    help="Directory for reports (default: outputs/)")
    ap.add_argument("--config", default=str(HERE / "pipeline_config.yaml"),
                    help="Pipeline settings YAML")
    ap.add_argument("--rules", help="Business rules YAML (overrides config)")
    ap.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    ap.add_argument("--contamination", type=float, help="Expected anomaly proportion")
    ap.add_argument("--quiet", action="store_true", help="Warnings and errors only")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)

        if args.output_dir:
            cfg["paths"]["output_dir"] = args.output_dir
        if args.rules:
            cfg["paths"]["rules_file"] = args.rules
        if args.no_figures:
            cfg["profiling"]["generate_figures"] = False
        if args.contamination is not None:
            cfg["validation"]["contamination"] = args.contamination
        if args.quiet:
            cfg["logging"]["verbose"] = False

        input_path = args.input or cfg["paths"].get("input")
        if not input_path:
            ap.error("no input given: pass --input or set paths.input in the config")

        summary = run_pipeline(input_path, cfg)

        fail_hard = cfg["thresholds"].get("fail_on_verdict_fail", True)
        sys.exit(1 if (summary["verdict"] == "FAIL" and fail_hard) else 0)

    except SchemaError as e:
        print(f"SCHEMA ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print(f"FILE NOT FOUND: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()