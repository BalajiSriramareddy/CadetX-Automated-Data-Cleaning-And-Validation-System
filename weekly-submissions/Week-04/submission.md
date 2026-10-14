# Week 04 — Submission

**Sprint dates:** _18 Jul 2026 → 25 Jul 2026_
**Scrum Master this week:** Balaji Sriramareddy (solo)

## What I did this week

Completed **Module 3 — Validation**, the component that decides whether a
dataset is safe to use downstream. Built across five components (one feature
branch + pull request each):

1. `rule_validation.py` — generic rule engine applying four check types per
   column: format (email / UK phone / UK postcode), numeric range, allowed
   category values, and required (not-null). Reports violation counts,
   percentages, and examples.
2. `config.py` + `rules.yaml` — the two-layer rules engine:
   - **Layer 1 (inferred):** default rules built automatically from Module 1's
     profiling report, so any dataset is validated with zero configuration.
   - **Layer 2 (declared):** an optional `rules.yaml` holding business rules that
     cannot be inferred from data (NPS must be 0–10, charges cannot be negative).
     Declared rules override inferred ones.
3. `anomaly_detection.py` — two complementary methods: IQR outliers (per-column,
   "is this value odd?") and Isolation Forest (row-level, "is this combination
   of values odd?"). The multivariate method catches records where no single
   value is extreme but the combination is implausible.
4. `health_score.py` — a 0–100 score with a PASS / WARNING / FAIL verdict.
   Violations are weighted by **severity**, not just counted: a missing required
   field (1.0) matters far more than a statistical anomaly (0.2), because an
   anomaly is unusual rather than necessarily wrong.
5. `validate.py` — the Validation API chaining all of the above into
   `validation_report.json`.

Also added `tests/test_module3.py` — **15 tests, all passing**. Two of these
verify the headline design claims directly: that the system still validates a
dataset with no profiling report and no rules file, and that declared business
rules correctly override inferred ones.

**Bug found and fixed:** the `rules.yaml` entry `allowed: [Yes, No]` was silently
parsed as `[True, False]`, so every valid row in the churn column was flagged as
a violation. This is YAML's boolean coercion behaviour (the "Norway problem",
where the country code NO becomes False). Fixed by quoting the values and adding
defensive coercion in the loader. A validation tool producing false positives on
good data is worse than one that misses errors, because it destroys trust in the
output.

## Meeting log

- **Team status:** continuing to complete the build independently, as recorded
  in earlier weeks and flagged to CadetX support.
- **Design note:** validation logic and rule *source* were deliberately kept
  separate — the rule engine only applies a spec, while the config layer decides
  what the rules are. This is what allows the same engine to run generically on
  any dataset and to apply domain rules when supplied.

## Progress against plan

- [x] Module 3 · rule-based validation engine
- [x] Module 3 · two-layer rules config (inferred + optional `rules.yaml`)
- [x] Module 3 · anomaly detection (IQR + Isolation Forest)
- [x] Module 3 · health scoring with PASS/WARNING/FAIL verdict
- [x] Module 3 · validation API (`validation_report.json`)
- [x] Module 3 · test suite (15 tests passing)
- [x] **Module 3 complete** — 39 tests now passing across Modules 1–3

## Blockers

- None this week.

## Next week

- Build **Module 4 — Pipeline Orchestration**: chain Modules 1 → 2 → 3 into a
  single command-line pipeline, add a minimal Dockerfile so the system runs
  anywhere, and begin final documentation (README and architecture diagram).