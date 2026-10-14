# Week 01 — Submission

**Sprint dates:** _27 Jun 2026 → 04 Jul 2026_ 
**Scrum Master this week:** Balaji Manjulamma Sriramareddy (volunteered to take the first rotation)

## What I did this week

Chose the project (Automated Data Cleaning & Validation System) and the dataset:
a synthetic UK rural fibre-broadband customer dataset with deliberately planted
data-quality issues, so the pipeline has real problems to detect.

Built **Module 1 — Advanced Data Profiling & Metadata Intelligence** in full,
across six components (one commit each):

1. `metadata.py` — type inference, semantic detection (email/phone/postcode/ID),
   mixed-type column detection.
2. `profiling_engine.py` — missing-value matrix, type-consistency check,
   cardinality analysis, correlation matrix, distributions.
3. `rule_engine.py` — PII detection, inconsistent-format detection, suspicious-
   column flagging (name + value triggered).
4. `visualisations.py` — backend-generated missing/correlation heatmaps,
   distribution and outlier plots (headless).
5. `profiling_api.py` + `schema.json` + module README — the Profiling API that
   assembles everything into `profiling_report.json`, plus documentation.
6. `demo.py` + `tests/test_module1.py` — end-to-end demo and 12 unit tests
   (all passing).

Key design decision: the module is **dataset-agnostic** — it profiles any CSV
with zero code changes. Verified by running the same code on the broadband
dataset and an unrelated retail-sales dataset; both produced valid reports.

Bug found and fixed: `profile.py` shadowed Python's standard-library `profile`
module, breaking imports. Renamed to `profiling_api.py`.

## Meeting log

- **Kickoff attempt:** proposed a first team meeting on Microsoft Teams and
  shared the project brief, repo, and dataset in the team channel.
- **Attendance:** the team has been largely unresponsive. One member has
  responded; the other(s) have not engaged despite follow-ups.
- **Decision:** proceeded with Module 1 solo to keep the project on schedule,
  while keeping the work modular (file-contract design) so any member who joins
  later can pick up a module without disruption.
- **Escalation:** flagged the team-unresponsiveness to CadetX support
  (support@cadetx.co.uk) so the record reflects the situation.

## Progress against plan

- [x] Dataset chosen and added to `data/raw/`
- [x] GitHub repo set up (Week 1–12 structure + project code)
- [x] Module 1 fully built, documented, and tested (produces `profiling_report.json`)
- [x] Dataset-agnostic design proven on two datasets

## Blockers

- Team largely unresponsive (one member engaged). Mitigation: proceeding solo
  with modular design; escalated to CadetX support. Will reassign a module to
  any member who becomes active.

## Next week

- Begin **Module 2 — Data Cleaning**: imputation, duplicate handling,
  normalisation, and before/after data-quality scoring, consuming the
  `profiling_report.json` from Module 1.