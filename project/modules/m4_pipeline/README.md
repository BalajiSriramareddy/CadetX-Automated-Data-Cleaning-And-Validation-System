# Module 4 — Integration, Orchestration & Production Engineering

One configurable pipeline that runs **profiling → cleaning → validation** with a
single command, packaged for production-style execution.

---

## 1. Architecture

```
                         ┌───────────────────────┐
                         │  pipeline_config.yaml │  how the pipeline runs
                         └───────────┬───────────┘
                                     │
   raw.csv ──────────────────────────▼──────────────────────────────┐
                            ┌─────────────────┐                     │
                            │  pipeline.py    │  orchestrator       │
                            │  (Module 4)     │                     │
                            └────────┬────────┘                     │
                                     │                              │
        ┌────────────────────────────┼────────────────────────────┐ │
        ▼                            ▼                            ▼ │
 ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
 │  MODULE 1   │            │  MODULE 2   │            │  MODULE 3   │
 │  Profiling  │───────────▶│  Cleaning   │───────────▶│ Validation  │
 └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
        │                          │                          │
        ▼                          ▼                          ▼
 profiling_report.json    cleaned_data.csv          validation_report.json
 + figures/*.png          cleaning_log.json         (health score + verdict)
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                                     │
                                     ▼
                          pipeline_summary.json
                        PASS · WARNING · FAIL   ──▶ exit code
```

**Cross-cutting components**

| Component | File | Responsibility |
|---|---|---|
| Orchestrator | `pipeline.py` | Runs the three stages in order, handles CLI |
| Configuration | `pipeline_config.yaml` | Pipeline settings (paths, toggles, thresholds) |
| Schemas | `schemas.py` | Contract-checks every stage's input and output |
| Logging | `logger.py` | INFO/WARNING/ERROR to console and `logs/*.log` |
| Container | `Dockerfile` | Reproducible execution environment |

**Two separate config files, deliberately:**
`pipeline_config.yaml` controls **how the pipeline runs** (paths, figures on/off,
contamination). `rules.yaml` (Module 3) declares **what counts as valid data**
(NPS 0–10). Tuning the pipeline never risks changing business rules.

---

## 2. Pipeline flow

| # | Stage | Reads | Writes | Key output |
|---|-------|-------|--------|------------|
| 0 | Load + input schema | `raw.csv` (as `dtype=str`) | — | validated DataFrame |
| 1 | **Profile** (M1) | DataFrame | `profiling_report.json`, `figures/` | types, PII, suspicious columns |
| 2 | **Clean** (M2) | DataFrame + profiling report | `cleaned_data.csv`, `cleaning_log.json` | quality before → after |
| 3 | **Validate** (M3) | cleaned data + rules | `validation_report.json` | health score + verdict |
| 4 | Summarise | all stage outputs | `pipeline_summary.json` | verdict + artefact paths |

**Why the CSV is read as `dtype=str`:** pandas' default type inference reads a
phone number like `07123456789` as an integer and silently drops the leading
zero. A data-quality tool must not let its own reader corrupt data before it has
been profiled. Types are inferred later by Module 1, where the decision is
visible and logged.

**Exit codes** (so the pipeline can gate an automated workflow):

| Code | Meaning |
|------|---------|
| `0` | Verdict PASS or WARNING — data is usable |
| `1` | Verdict FAIL — do not use downstream |
| `2` | Pipeline error (bad input, broken contract, missing file) |

---

## 3. API usage

### Command line

```bash
# simplest run
python modules/m4_pipeline/pipeline.py --input data/raw/broadband_customers.csv

# with business rules and a custom output directory
python modules/m4_pipeline/pipeline.py \
    --input data/raw/broadband_customers.csv \
    --rules modules/m3_validation/rules.yaml \
    --output results/

# faster run for iteration; quieter logs; stricter anomaly threshold
python modules/m4_pipeline/pipeline.py --input data.csv \
    --no-figures --quiet --contamination 0.1

# drive everything from the config file
python modules/m4_pipeline/pipeline.py --config modules/m4_pipeline/pipeline_config.yaml
```

| Flag | Effect |
|------|--------|
| `--input` | Path to the raw CSV (required unless set in config) |
| `--output` / `--output-dir` | Where reports are written |
| `--config` | Pipeline settings YAML |
| `--rules` | Business rules YAML (Module 3) |
| `--no-figures` | Skip PNG generation (faster) |
| `--contamination` | Expected anomaly proportion (Isolation Forest) |
| `--quiet` | Console shows warnings and errors only |

Precedence: **CLI flags → config file → built-in defaults.**

### Python

```python
from pipeline import run_pipeline, load_config

config = load_config("modules/m4_pipeline/pipeline_config.yaml")
config["profiling"]["generate_figures"] = False

summary = run_pipeline("data/raw/broadband_customers.csv", config)
print(summary["verdict"], summary["health_score"])
```

### Docker

```bash
docker build -t data-quality-pipeline .

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  data-quality-pipeline --input data/raw/broadband_customers.csv
```

---

## 4. Schemas (stage contracts)

| Stage | File | Required top-level keys |
|-------|------|-------------------------|
| Input | `raw.csv` | ≥1 row, ≥1 named column, no duplicate column names |
| Profiling | `profiling_report.json` | `dataset`, `generated_at`, `metadata`, `profiling`, `rules` |
| Cleaning | `cleaning_log.json` | `generated_at`, `rows_in`, `rows_out`, `quality`, `steps` |
| Validation | `validation_report.json` | `dataset`, `generated_at`, `rules_applied`, `rule_validation`, `anomaly_detection`, `health` |
| Summary | `pipeline_summary.json` | `input`, `rows_in`, `rows_out`, `verdict`, `outputs` |

Each stage is checked immediately after it runs, so a broken contract fails
where it happened rather than three stages later.

---

## 5. Logging

Two destinations at once: readable progress on the console, and a permanent
audit trail in `logs/pipeline_<timestamp>.log`.

| Level | Used for | Example |
|-------|----------|---------|
| INFO | Normal progress | `rows 420 -> 400 \| quality 81.69 -> 93.08` |
| WARNING | Real but non-fatal issues | `column 'email': 32.1% of values fail email format` |
| ERROR | Stage or verdict failure | `VERDICT: FAIL — do not use downstream` |

Warnings are generated from the profiling findings themselves, so the log
surfaces genuine data problems rather than just narrating execution.

---

## 6. Data versioning

Git LFS, configured in `.gitattributes`: CSV, Parquet, XLSX and PNG files are
stored as LFS objects; JSON reports stay as normal Git objects because they are
text and diff usefully between runs.

```bash
git lfs install
git add .gitattributes
git commit -m "Track data files with Git LFS"
```

**Git LFS rather than DVC:** DVC needs a separate remote (S3, Google Drive) to be
useful, whereas LFS works directly with GitHub and needs no extra service. For a
project of this size that is the proportionate choice; DVC would be the upgrade
if datasets grew or experiments needed reproducing against specific data versions.

---

## 7. Testing

```bash
pytest tests/ -v                     # all modules
pytest tests/test_module4.py -v      # integration only
```

| Suite | Tests | Covers |
|-------|-------|--------|
| `test_module1.py` | 12 | Profiling: types, PII, report structure |
| `test_module2.py` | 12 | Cleaning: imputation, dedup, normalisation, scoring |
| `test_module3.py` | 15 | Validation: rules, config layers, anomalies, health |
| `test_module4.py` | 16 | **Integration**: end-to-end, schemas, config, logging |
| **Total** | **55** | |

Two integration tests are regression locks for bugs that only appeared once the
modules ran together:

- `test_leading_zeros_survive_the_whole_pipeline` — `read_csv` was coercing
  phone numbers to integers and dropping the leading zero.
- `test_text_column_not_destroyed_by_one_numeric_value` — a postcode column
  containing `12345` was being coerced to numeric, turning every real postcode
  into `NaN`.

Both passed their own module's unit tests. Only integration testing caught them.

---

## 8. Deployment

`Dockerfile` + `requirements.txt` produce a reproducible image. Dependencies are
installed in a separate layer before the application code is copied, so changing
code does not force a dependency reinstall on rebuild.

CI (GitHub Actions) runs the test suite on every push — see
`.github/workflows/ci.yml`.

---

> ⚠️ All project data is synthetic. No real customers.