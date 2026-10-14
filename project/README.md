# Project code

This folder holds the pipeline itself. The overview, results and design decisions are in the [main README](../README.md).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Full pipeline:

```bash
python modules/m4_pipeline/pipeline.py --input data/raw/broadband_customers.csv --output outputs/
```

| Flag | What it does |
|---|---|
| `--input` | Path to the raw CSV |
| `--output` | Folder for the JSON reports (default `outputs/`) |
| `--config` | Settings file (default `modules/m4_pipeline/pipeline_config.yaml`) |
| `--rules` | Business rules YAML, overriding the config |
| `--contamination` | Expected share of anomalies for Isolation Forest |
| `--no-figures` | Skip the PNG charts |
| `--quiet` | Show warnings and errors only |

Exit codes: 0 for PASS or WARNING, 1 for FAIL, 2 for a pipeline error.

Profiling demo on two unrelated datasets, to show the profiler needs no code changes between them:

```bash
python demo.py
```

Tests:

```bash
pytest tests/ -v
```

To rebuild the synthetic dataset and its planted issues:

```bash
python data/generate_dataset.py
```

## Outputs

| File | Written by | Contents |
|---|---|---|
| `outputs/profiling_report.json` | Module 1 | Types, missing values, cardinality, correlation, personal-data flags |
| `data/processed/cleaned_data.csv` | Module 2 | The cleaned dataset |
| `outputs/cleaning_log.json` | Module 2 | Every change made, plus the before and after quality score |
| `outputs/validation_report.json` | Module 3 | Rule violations, anomalies, health score |
| `outputs/pipeline_summary.json` | Module 4 | One-page summary and verdict |
| `logs/pipeline_<timestamp>.log` | Module 4 | Run log |

`outputs/`, `logs/` and `data/processed/` are ignored by Git, since they are regenerated on every run.

## Modules

| Folder | Main entry point | Other files |
|---|---|---|
| `modules/m1_profiling` | `profiling_api.py` | `metadata.py`, `profiling_engine.py`, `rule_engine.py`, `visualisations.py` |
| `modules/m2_cleaning` | `clean.py` | `deduplication.py`, `normalisation.py`, `imputation.py`, `ml_imputation.py`, `fuzzy_matching.py`, `schema_inference.py`, `transformation.py`, `quality_score.py` |
| `modules/m3_validation` | `validate.py` | `config.py`, `rules.yaml`, `rule_validation.py`, `anomaly_detection.py`, `health_score.py`, plus the optional `advanced_anomaly.py`, `nlp_classifier.py`, `ai_error_detection.py`, `evaluation.py` |
| `modules/m4_pipeline` | `pipeline.py` | `pipeline_config.yaml`, `schemas.py`, `logger.py` |

Modules 1 and 4 have their own notes: [m1_profiling/README.md](modules/m1_profiling/README.md) and [m4_pipeline/README.md](modules/m4_pipeline/README.md).
