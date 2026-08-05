# Week 05 — Submission

**Sprint dates:** _25 Jul 2026 → 01 Aug 2026_
**Scrum Master this week:** Balaji Sriramareddy (solo)

## What I did this week

**Completed Module 2 in full**, closing every remaining item from the brief
(one feature branch + pull request each):

1. `schema_inference.py` — infers each column's expected type, expected range,
   and expected format, producing a schema that cleaning and validation check
   against. Distinct from Module 1's profiling: profiling describes what the
   data IS, this defines what it SHOULD be.
2. `fuzzy_matching.py` — flags candidate near-duplicate values (e.g. "Jon
   Smith" vs "John Smith") in name/free-text columns for human review. This
   never auto-merges or deletes: unlike exact/normalised deduplication, a
   fuzzy match is a judgement call, and a wrong auto-merge would destroy a
   real customer's record.
3. `ml_imputation.py` — KNN, regression, and iterative (MICE-style) imputers,
   offered as an optional upgrade to the statistical (median/mode) default.
   Documented explicitly why they are optional: they need multiple numeric
   columns, they are harder to explain to a stakeholder than "the median",
   and they can quietly encode bias from correlated columns.
4. `transformation.py` — scaling, one-hot encoding, and feature extraction
   (date parts, ratio features), run after cleaning to prepare data for
   modelling without overwriting the human-readable cleaned columns.

**Began Module 3's advanced components:**

5. `advanced_anomaly.py` — Local Outlier Factor (catches locally-unusual
   points that a global method like Isolation Forest can miss) and a
   lightweight autoencoder built on scikit-learn's MLPRegressor rather than
   TensorFlow/PyTorch, to avoid an unjustified heavy dependency for one
   component.

**Verified against the full project brief:** built a line-by-line tracker
against every listed requirement across all four modules, to make sure nothing
from the original brief was skipped or quietly descoped without a record.

**Bug found and fixed:** the transformation engine's category-encoding step
initially caught date columns as if they were categories, which would have
one-hot encoded a date into one column per unique value — meaningless. Fixed
by routing dates to proper date-feature extraction instead, and confirmed with
a direct test rather than assuming the fix worked.

## Meeting log

- Continued solo, as recorded in prior weeks; escalation to CadetX support
  remains on record from earlier submissions.

## Progress against plan

- [x] Module 2 — schema inference engine
- [x] Module 2 — fuzzy duplicate matching
- [x] Module 2 — ML-based imputation (KNN, regression, iterative)
- [x] Module 2 — transformation engine (scaling, encoding, feature extraction)
- [x] **Module 2 now 100% complete against the brief**
- [x] Module 3 — Local Outlier Factor
- [x] Module 3 — autoencoder (lightweight)
- [ ] Module 3 — NLP column classifier
- [ ] Module 3 — AI-based error detection (impossible values, patterns, drift)
- [ ] Module 3 — evaluation (precision/recall, confusion matrix, ROC)

## Blockers

- None. Proceeding solo, working through a full requirements tracker to
  ensure completeness against the original brief.

## Next week

- Finish Module 3's remaining advanced components (NLP column classifier,
  AI error detection, evaluation metrics), then move to per-module and final
  presentation materials (architecture diagram, working demo, results).