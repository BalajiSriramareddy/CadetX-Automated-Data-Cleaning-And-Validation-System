# Week 02 — Submission

**Sprint dates:** _04 Jul 2026 → 11 Jul 2026_
**Scrum Master this week:** Balaji Sriramareddy (solo — see note below)

## What I did this week

Began **Module 2 — Data Cleaning**, which consumes the `profiling_report.json`
from Module 1 and acts on what it found. Two components built, tested, and
merged this week (one feature branch + pull request each):

1. `imputation.py` — fills missing values by type: numeric columns with the
   **median** (robust to outliers), categorical with the **mode**. Every fill is
   written to an audit log so changes are fully traceable. Driven by Module 1's
   inferred types, so it stays dataset-agnostic.
2. `deduplication.py` — removes exact and normalised duplicates (e.g.
   "Ava Patel" vs "  ava patel "), and **flags** key conflicts (same identifier,
   different data) for review rather than deleting them.

Adopted a professional Git workflow this week: `main` kept stable, each
component built on its own feature branch, merged via a reviewed pull request.

**Bug found and fixed:** deduplication silently missed duplicates because the
code checked `dtype == object`, but this pandas version reports text columns as
`str` — so normalisation was being skipped. Caught it via a known-answer test
(the duplicate count was wrong) and fixed it to test "not numeric/datetime"
instead, which is robust across pandas versions.

## Meeting log

- **Team status:** team membership has kept changing and engagement has been
  inconsistent throughout the sprint.
- **Decision:** to keep the project on schedule and to a consistent standard, I
  have decided to complete the build independently, while keeping the modular
  file-contract design so any future contributor could still pick up a module.
- **Action:** recorded this decision and notified CadetX support so the position
  is on record. _(Confirm this notification has actually been sent.)_

## Progress against plan

- [x] Module 2 · imputation (median/mode + audit log) — built, tested, merged
- [x] Module 2 · deduplication (exact/normalised + key-conflict flagging) — built, tested, merged
- [x] Feature-branch + pull-request workflow established
- [ ] Module 2 · normalisation (fix mixed-type £ columns, standardise dates/categories)
- [ ] Module 2 · quality scoring (before/after metric)
- [ ] Module 2 · cleaning API (`clean.py` → `cleaned_data.csv` + `cleaning_log.json`)

## Blockers

- Inconsistent team engagement. Mitigation: proceeding solo with modular design;
  decision recorded and flagged to CadetX support.

## Next week

- Complete **Module 2**: build `normalisation.py`, `quality_score.py`, and the
  `clean.py` cleaning API, producing `cleaned_data.csv` and `cleaning_log.json`
  with a measurable before/after quality score.