# Module 1 — Advanced Data Profiling & Metadata Intelligence

Understands a dataset **before** any cleaning or ML happens: extracts metadata,
profiles the data in depth, generates backend visualisations, and flags
suspicious or sensitive columns.

**Output:** `profiling_report.json` (schema in [`schema.json`](schema.json))
**Design:** dataset-agnostic — runs on any CSV with zero code changes.

---

## Files

| File | Responsibility |
|------|----------------|
| `metadata.py` | Type inference, semantic detection (email/phone/date/id), mixed-type detection |
| `profiling_engine.py` | Missing-value matrix, type-consistency, cardinality, correlation, distributions |
| `rule_engine.py` | PII detection, inconsistent-format detection, suspicious-column flagging |
| `visualisations.py` | Backend PNGs: missing + correlation heatmaps, distributions, outlier boxplots |
| `profiling_api.py` | **Public API** — wires the above into `profiling_report.json` |

Dependency direction: `profiling_api.py` → (`metadata`, `profiling_engine`,
`rule_engine`, `visualisations`); `profiling_engine` and `rule_engine` reuse
`metadata`. Type logic lives in one place (`metadata.py`).

---

## How to run

```bash
# CLI
python profiling_api.py --input data/raw/broadband_customers.csv
python profiling_api.py --input data/raw/any.csv --no-figures   # skip PNGs
```

```python
# As a library
import pandas as pd
from profiling_api import build_profiling_report

df = pd.read_csv("data/raw/broadband_customers.csv")
report = build_profiling_report(df, source="broadband", with_figures=True)
```

---

## Key function signatures

```python
# metadata.py
extract_metadata(df: DataFrame) -> dict          # per-column metadata
infer_type(series: Series) -> str
detect_semantic_type(series: Series, column_name: str = "") -> str
detect_mixed_types(series: Series) -> dict

# profiling_engine.py
profile_data(df: DataFrame) -> dict              # missing, consistency, cardinality, corr, dist

# rule_engine.py
run_rules(df: DataFrame) -> dict                 # pii, formats, suspicious
expected_type(series: Series, column_name: str) -> str

# visualisations.py
generate_all(df: DataFrame, output_dir="outputs/figures") -> dict[str, str | None]

# profiling_api.py  (public API)
build_profiling_report(df, source="", with_figures=True, figures_dir=None) -> dict
```

---

## Integration instructions

- **Downstream (Module 2 — Cleaning):** read `profiling_report.json`. Use
  `metadata[col].inferred_type` to choose an imputation strategy, and
  `rules.suspicious_columns` / `rules.inconsistent_formats` to target cleaning.
- **Contract:** the report always contains `dataset`, `metadata`, `profiling`,
  and `rules`. `figures` may be empty if run with `--no-figures`. Validate against
  `schema.json`.
- **Tuning:** detection thresholds live at the top of each file
  (`MATCH_THRESHOLD`, `HIGH_MISSING_PCT`, `HIGH_CARDINALITY`). Adjust there, not
  inline.

---

>  All project data is synthetic. No real customers.
