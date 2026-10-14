# Automated Data Cleaning and Validation System

A Python pipeline that checks a dataset before anyone builds a report or model on it. It profiles the data, cleans it, validates it against rules, and returns a 0 to 100 health score with a PASS, WARNING or FAIL verdict. One command runs the whole thing.

I built it during the CadetX virtual work experience programme (June to August 2026). The brief was written for a team of three; the team didn't engage, so I built all four modules myself and kept the design modular so anyone joining later could pick up a module.

## What it does on the project dataset

The test data is a synthetic UK rural fibre-broadband customer file: 1,030 rows, 16 columns, with known problems planted on purpose (duplicates, invalid emails and postcodes, NPS scores outside 0 to 10, negative values, prices stored as text like "£44.99", mixed date formats, inconsistent churn labels).

Running the pipeline on it gives:

| Measure | Result |
|---|---:|
| Rows in / rows out (30 exact duplicates removed) | 1,030 / 1,000 |
| Data quality score before cleaning | 87.4 |
| Data quality score after cleaning | 96.3 |
| Rule violations found at validation | 267 across 6 columns |
| Health score and verdict | 81.0, WARNING |

It also flags six columns as likely personal data (name, email, phone, postcode, customer ID, payment method) and reports the install date column as using more than one date format. The full run takes about a second.

The test suite has 89 tests across the four modules, and all of them pass.

## How it works

Each module writes a file and the next module reads it, so every stage can be run and tested on its own.

```
raw CSV
  -> M1 Profile   -> profiling_report.json
  -> M2 Clean     -> cleaned_data.csv + cleaning_log.json
  -> M3 Validate  -> validation_report.json
  -> M4 Pipeline  -> pipeline_summary.json (score + verdict)
```

Module 1 (profiling) infers each column's type and meaning (email, phone, postcode, ID), measures missing values, cardinality and correlation, detects columns holding mixed types, and flags personal data. It works on any CSV without code changes; I checked this by running the same code on an unrelated retail sales file.

Module 2 (cleaning) removes exact and normalised duplicates, converts text prices to numbers, puts dates into one format, fills missing values (median for numbers, mode for categories, with KNN and iterative imputation as options), and scores quality before and after. Possible near-duplicate names such as "Jon Smith" and "John Smith" are flagged for a person to review and never merged automatically.

Module 3 (validation) runs two layers of rules. The first layer is inferred from the profiling report, so any dataset can be validated with no setup. The second is an optional `rules.yaml` for business rules the data can't reveal, such as NPS having to sit between 0 and 10. Anomaly detection uses IQR outliers per column and Isolation Forest across whole rows, which catches records where no single value is extreme but the combination is implausible. Local Outlier Factor, a small autoencoder, a column classifier and drift checks are available as optional extras, switched off by default. Violations are weighted by severity: a missing required field counts five times as much as a statistical outlier.

Module 4 (pipeline) chains the three modules, reads its settings from `pipeline_config.yaml` (command-line flags override it), checks each stage's output against a schema, logs to file, and returns exit code 0 for PASS or WARNING, 1 for FAIL and 2 for an error. That last part means it can sit in an automated workflow as a quality gate.

## Bugs found by testing

These only showed up because the tests checked against known answers:

- Deduplication missed duplicates because the code checked for `dtype == object`, and the installed pandas version reports text columns differently. It now tests for "not numeric or date" instead.
- Parsing dates with `dayfirst=True` silently swapped day and month on ISO dates (2024-03-10 became 2024-10-03). ISO and UK-style dates are now parsed separately.
- Phone numbers stored as text were being converted to numbers, which dropped the leading zero. Identifier columns are now never converted.
- `allowed: [Yes, No]` in `rules.yaml` was read by YAML as `[True, False]`, so every valid churn value failed validation. The values are now quoted.

## Run it

```bash
git clone https://github.com/BalajiSriramareddy/CadetX-Automated-Data-Cleaning-And-Validation-System.git
cd CadetX-Automated-Data-Cleaning-And-Validation-System/project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python modules/m4_pipeline/pipeline.py --input data/raw/broadband_customers.csv --output outputs/
pytest tests/ -v
```

Useful flags: `--rules` to use your own rules file, `--no-figures` to skip the charts, `--quiet` to show only warnings and errors. Reports go to `outputs/` and the cleaned file to `data/processed/`.

With Docker:

```bash
cd project
docker build -t data-quality-pipeline .
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" \
  data-quality-pipeline --input data/raw/broadband_customers.csv
```

GitHub Actions runs the tests and a pipeline smoke test on every push to `main`.

## Repository layout

```
.github/workflows/ci.yml     tests on every push
project/
  data/raw/                  broadband_customers.csv (synthetic)
  data/generate_dataset.py   rebuilds the dataset and its planted issues
  modules/m1_profiling/
  modules/m2_cleaning/
  modules/m3_validation/     includes rules.yaml
  modules/m4_pipeline/       pipeline.py, pipeline_config.yaml
  tests/                     89 pytest tests
  Dockerfile
weekly-submissions/          CadetX sprint reports, weeks 1 to 5
```

More detail on the individual modules is in [project/README.md](https://github.com/BalajiSriramareddy/CadetX-Automated-Data-Cleaning-And-Validation-System/blob/main/project/README.md), the [profiling module notes](https://github.com/BalajiSriramareddy/CadetX-Automated-Data-Cleaning-And-Validation-System/blob/main/project/modules/m1_profiling/README.md) and the [pipeline module notes](https://github.com/BalajiSriramareddy/CadetX-Automated-Data-Cleaning-And-Validation-System/blob/main/project/modules/m4_pipeline/README.md).

## Tools

Python (pandas, NumPy, scikit-learn, matplotlib, seaborn, PyYAML), pytest, Docker, GitHub Actions, Git with a feature-branch and pull-request workflow.

All data in this repository is synthetic. There are no real customers in it.

Balaji Manjulamma Sriramareddy · [LinkedIn](https://www.linkedin.com/in/balaji-sriramareddy/) · [Portfolio](https://balajisriramareddy.github.io/)
