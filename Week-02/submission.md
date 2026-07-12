# Week 02 — Submission

**Sprint dates:** _04 Jul 2026 → 11 Jul 2026_
**Scrum Master this week:** Adib Hassan— AI/ML

## What I did this week

Joined the project as the AI/ML contributor and was added as a GitHub
collaborator on the repo. Completed the remaining scope of
**Module 2 — Data Cleaning & Transformation**, building on top of the
`imputation.py` engine that was already merged (PR #1):

1. **`drop_duplicates()`** — exact duplicate row detection and removal,
   with a before/after row-count log entry.
2. **`normalise()`** — standardises formats using Module 1's
   `profiling_report.json` findings, driven by column metadata rather
   than hardcoded column names (dataset-agnostic, consistent with
   Module 1's design principle):
   - currency-as-text (`"£68.48"`) → float
   - mixed date formats (`"2024-12-05"` / `"Nov 2023"`) → one consistent
     `datetime64` dtype
   - inconsistent Yes/No-style categorical spellings
     (`Yes`/`yes`/`Y`/`1`) → one consistent label
   - clearly-invalid identifier placeholders (e.g. junk phone values)
     flagged for manual review rather than silently guessed at
3. **`quality_score()`** — a 0–100 completeness + uniqueness score,
   computed before and after cleaning to report a quantified delta.
4. **`main()`** — wired the full pipeline together per the file contract:
   `data/raw/*.csv` + `outputs/profiling_report.json` in,
   `data/processed/cleaned_data.csv` + `outputs/cleaning_log.json` out.
5. **`tests/test_module2.py`** — 7 unit tests (known-answer style, matching
   Module 1's test conventions). All passing.

**Bug found and fixed:** Module 1 flags the `phone` column as
`inferred_type: "numeric_as_text"` (because most values look like digit
strings), but its `semantic_type` is `"phone"`. An early version of
`normalise()` matched on `inferred_type` alone and coerced phone numbers
to float, destroying leading zeros and identifier meaning. Fixed by
requiring `semantic_type == "numeric"` before applying the currency-strip
conversion. Added a regression test
(`test_normalise_does_not_touch_phone_even_if_numeric_as_text`) so this
can't silently reappear.

**Verified end-to-end** on the real dataset (1,030 rows, 16 columns):

| | Before | After |
|---|---|---|
| Quality score | 98.36 | 100.0 |
| Exact duplicate rows | 30 | 0 |
| Missing values | 236 | 0 |

## Meeting log

- **Access setup:** requested and received collaborator access to the
  team repo; first `git push` was initially rejected (403 — permission
  not yet granted), resolved once added as a collaborator.
- **Branching:** created a dedicated branch (`feature/m2-cleaning-full`)
  separate from the already-merged `feature/m2-imputation` branch, to
  keep PR history clean — one PR per logical unit of work.
- **No live sync meeting held this week** [confirm/adjust]; coordination
  happened asynchronously via the repo (existing `imputation.py` was read
  and reused rather than duplicated, to avoid conflicting with prior work).

## Progress against plan

- [x] Reviewed existing `imputation.py` (Module 2, partially built) before
      writing any new code, to avoid duplicating work
- [x] Implemented `drop_duplicates()`, `normalise()`, completed `main()`
- [x] Added `tests/test_module2.py` (7/7 passing)
- [x] Ran the full Module 1 → Module 2 pipeline end-to-end on the real
      dataset and confirmed the quality-score delta
- [x] Opened PR from `feature/m2-cleaning-full` → `main`

## Blockers

- GitHub collaborator permission was not granted at the start of the
  week, blocking the first push attempt (403 error). Resolved once
  Balaji added me as a collaborator.
- PR is open and **awaiting review/merge** from the team.

## Next week

- Address any review feedback on the Module 2 PR.
- Sync with the team on **Module 3 — Validation** (rule-based checks +
  anomaly detection), which will consume `cleaned_data.csv` and
  `cleaning_log.json` from this module.
- Revisit the `quality_score()` formula (currently completeness +
  uniqueness only) to optionally fold in format-validity from the
  profiling report, per the Week 7 refinement note in the project plan.