# Week 03 — Submission

**Sprint dates:** _11 Jul 2026 → 18 Jul 2026_
**Scrum Master this week:** Balaji Sriramareddy (solo)

## What I did this week

Completed **Module 2 — Data Cleaning**. Building on last week's imputation and
deduplication, I finished the remaining components (one feature branch + pull
request each) and added a full test suite:

1. `normalisation.py` — standardises formats: coerces currency-as-text to real
   numbers (£44.99 → 44.99), parses mixed date formats to one ISO standard,
   unifies Yes/No-style values, and canonicalises category spelling/case.
2. `quality_score.py` — a 0–100 data-quality score across three measurable
   dimensions (completeness, uniqueness, type validity), plus a before/after
   delta that proves the cleaning worked.
3. `clean.py` — the Cleaning API: chains deduplication → normalisation →
   imputation → scoring and writes `cleaned_data.csv` + `cleaning_log.json`.
4. `tests/test_module2.py` — 12 unit + integration tests, all passing.

End-to-end result on a messy sample: rows reduced correctly (duplicates
removed), all missing values handled, mixed-type columns fixed, and the quality
score rose measurably before → after.

**Two bugs found and fixed via testing:**
- *ISO date corruption:* a single global `dayfirst=True` was silently flipping
  month/day on ISO dates (2024-03-10 → 2024-10-03). Fixed by parsing ISO and
  UK-style dates separately. Locked in with a regression test.
- *Identifier coercion:* text-stored identifiers (e.g. phone numbers) were being
  coerced to floats, destroying leading zeros (07123… → 7123…). Added a guard so
  phone/postcode/identifier columns are never converted to numbers. Tested.

## Meeting log

- **Team status:** continued to complete the build independently.
- **Design note:** kept the modular, file-contract structure (separate
  imputation / deduplication / normalisation / scoring modules), so each piece is
  independently testable and any future contributor could pick up a module.
- **Escalation:** solo position previously recorded with CadetX support.

## Progress against plan

- [x] Module 2 · normalisation (numeric/date/category, with identifier guard)
- [x] Module 2 · quality scoring (before/after delta)
- [x] Module 2 · cleaning API (`clean.py` → `cleaned_data.csv` + `cleaning_log.json`)
- [x] Module 2 · test suite (12 tests passing)
- [x] **Module 2 complete**

## Blockers

- None this week. Proceeding solo as planned.

## Next week

- Begin **Module 3 — Validation**: a generic rule engine (format, range, and
  outlier checks) plus an optional `rules.yaml` config for domain rules — the
  feature that lets the system validate any dataset out of the box and apply
  business rules when supplied. Output: `validation_report.json`.