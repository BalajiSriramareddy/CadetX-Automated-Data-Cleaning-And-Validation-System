"""
MODULE 3 · FILE 4 — VALIDATION HEALTH SCORE
===========================================
Turns rule violations and anomalies into ONE interpretable 0-100 score, plus a
plain-English verdict.

KEY IDEA: violations are not equally serious, so they are weighted by SEVERITY
rather than simply counted.

  critical (1.0)  required-field missing        -> the record is unusable
  high     (0.7)  format failure                -> value is wrong (bad email/postcode)
  high     (0.7)  disallowed category           -> value outside the permitted set
  medium   (0.5)  out-of-range numeric          -> value is impossible per business rule
  low      (0.2)  statistical outlier / anomaly -> unusual, but possibly legitimate

Why weight anomalies lowest? Because an anomaly is not necessarily an error. A
customer on an unusually high tariff is odd, not wrong. A missing customer ID
genuinely breaks the record. Treating them the same would make the score
meaningless — and would train people to ignore it.

Verdict thresholds (tunable):
  >= 90  PASS      -> safe for downstream use
  70-89  WARNING   -> usable, but issues need review
  <  70  FAIL      -> do not use until fixed

Public entry point:  compute_health_score(validation_results, anomaly_results, n_rows)
"""
from __future__ import annotations

SEVERITY_WEIGHTS = {
    "required": 1.0,
    "format": 0.7,
    "allowed_values": 0.7,
    "range": 0.5,
    "anomaly": 0.2,
}

PASS_THRESHOLD = 90
WARN_THRESHOLD = 70


def _severity_of(rule_name: str) -> tuple[str, float]:
    """Map a rule label (e.g. 'format:email', 'range:[0,10]') to its severity."""
    key = rule_name.split(":")[0]
    if key in SEVERITY_WEIGHTS:
        return key, SEVERITY_WEIGHTS[key]
    return "anomaly", SEVERITY_WEIGHTS["anomaly"]


def compute_health_score(validation_results: dict,
                         anomaly_results: dict | None = None,
                         n_rows: int = 0) -> dict:
    """Score the dataset's validation health.

    The score is 100 minus the weighted penalty, where each violation's penalty
    is (violations / n_rows) * severity_weight, expressed as a percentage.
    """
    if n_rows <= 0:
        return {"health_score": 0.0, "verdict": "UNKNOWN",
                "reason": "no rows to score"}

    penalty = 0.0
    breakdown: dict[str, dict] = {}

    # --- rule violations ---
    for col, checks in (validation_results or {}).get("column_violations", {}).items():
        for check in checks:
            severity, weight = _severity_of(check["rule"])
            rate = check["violations"] / n_rows
            contribution = rate * weight * 100
            penalty += contribution
            entry = breakdown.setdefault(severity, {"violations": 0, "penalty": 0.0,
                                                    "columns": []})
            entry["violations"] += check["violations"]
            entry["penalty"] = round(entry["penalty"] + contribution, 2)
            if col not in entry["columns"]:
                entry["columns"].append(col)

    # --- anomalies (lowest severity) ---
    if anomaly_results:
        mv = anomaly_results.get("multivariate", {})
        n_anom = mv.get("anomalies_found", 0) or 0
        iqr_total = sum(v.get("outliers", 0)
                        for v in (anomaly_results.get("iqr_outliers") or {}).values())
        total_anom = n_anom + iqr_total
        if total_anom:
            contribution = (total_anom / n_rows) * SEVERITY_WEIGHTS["anomaly"] * 100
            penalty += contribution
            breakdown["anomaly"] = {
                "violations": int(total_anom),
                "penalty": round(contribution, 2),
                "columns": list((anomaly_results.get("iqr_outliers") or {}).keys()),
            }

    score = max(0.0, round(100 - penalty, 2))

    if score >= PASS_THRESHOLD:
        verdict, reason = "PASS", "Data quality is acceptable for downstream use."
    elif score >= WARN_THRESHOLD:
        verdict, reason = "WARNING", "Usable, but issues should be reviewed before use."
    else:
        verdict, reason = "FAIL", "Do not use downstream until issues are resolved."

    return {
        "health_score": score,
        "verdict": verdict,
        "reason": reason,
        "total_penalty": round(penalty, 2),
        "severity_breakdown": breakdown,
        "thresholds": {"pass": PASS_THRESHOLD, "warning": WARN_THRESHOLD},
        "severity_weights": SEVERITY_WEIGHTS,
    }