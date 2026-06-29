# CadetX Project — Automated Data Quality & Validation System

CadetX Virtual Work Experience submission.
**Intern:** Balaji Manjulamma Sriramareddy

This repository holds both the **weekly programme submissions** (`Week-01` …
`Week-12`) and the **project code** (`project/`).

---

## Repository structure

```
CadetX-Broadband-Data-Quality/
├── Week-01/ … Week-12/     Weekly submissions: sprint summary, meeting log, progress
│   └── submission.md       (template — fill in each week)
└── project/                The actual pipeline (code, data, modules)
    ├── README.md           Full project documentation
    ├── data/               Synthetic broadband customer dataset + generator
    ├── modules/            M1 Profiling · M2 Cleaning · M3 Validation · M4 Pipeline
    ├── outputs/            JSON reports
    └── tests/
```

> **Why this layout:** the CadetX programme requires Week 1–12 folders for weekly
> submissions; the project brief requires a module-based pipeline. This repo serves
> both — weekly notes live in the week folders, the living code lives in `project/`.

---

## The project in one line

A backend "data gatekeeper" that profiles, cleans, validates, and scores any
dataset before it is used — so unreliable data never drives faulty insights.
Built on a **synthetic UK rural fibre-broadband customer dataset**. Full details
in [`project/README.md`](project/README.md).

---

## 12-week plan (build → harden → present)

The core build is front-loaded; later weeks harden, document, and extend.

| Week | Focus |
|------|-------|
| 1 | Setup, dataset, **Module 1 — Profiling** |
| 2 | **Module 2 — Cleaning** (imputation, dedup, quality score) |
| 3 | **Module 3 — Validation** (rule-based + anomaly detection) |
| 4 | **Module 4 — Pipeline** (one-command run + Docker) |
| 5 | README, architecture diagram, tests |
| 6 | End-to-end testing + bug fixing |
| 7 | Refine quality-scoring logic; expand validation rules |
| 8 | "Future work" extensions (optional ML methods) |
| 9 | Documentation polish; per-module write-ups |
| 10 | Final demo build + dry run |
| 11 | Presentation prep |
| 12 | Final presentation + submission |

> Each week, fill in `Week-XX/submission.md` with what was done, the meeting log,
> and blockers. Consistent weekly updates are graded.

---

> ⚠️ All data in this project is **synthetic**. No real customers.
