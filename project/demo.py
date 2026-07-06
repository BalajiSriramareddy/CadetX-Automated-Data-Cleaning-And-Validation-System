"""
MODULE 1 · FILE 6a — END-TO-END DEMO
====================================
Runnable demonstration of Module 1. Two purposes:
  1. Show the full profiling pipeline on the broadband dataset.
  2. PROVE it is dataset-agnostic by running the SAME code, zero changes,
     on a completely unrelated dataset.

Run:  python demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent / "modules" / "m1_profiling"))
from profiling_api import build_profiling_report  # noqa: E402


def summarise(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"DATASET: {name}   ({len(df)} rows x {df.shape[1]} cols)")
    print("=" * 60)
    report = build_profiling_report(df, source=name, with_figures=False)

    pii = list(report["rules"]["potential_pii"].keys())
    suspicious = report["rules"]["suspicious_columns"]
    formats = report["rules"]["inconsistent_formats"]

    print(f"PII columns flagged:      {pii or 'none'}")
    print(f"Suspicious columns:       {list(suspicious.keys()) or 'none'}")
    print(f"Format issues found in:   {list(formats.keys()) or 'none'}")

    # show a couple of metadata inferences
    print("Sample type inferences:")
    for col in list(df.columns)[:4]:
        m = report["metadata"][col]
        print(f"   {col:16} -> {m['inferred_type']:16} ({m['semantic_type']})")


def main():
    # --- Dataset 1: broadband customers (the project dataset) ---
    broadband = pd.DataFrame({
        "customer_id": range(10000, 10010),
        "email": ["a@x.com", "bad-email", "c@y.co.uk", "d@z.com", "no-at",
                  "f@x.com", "g@y.com", "h@z.co.uk", "i@x.com", "j@y.com"],
        "postcode": ["OX7 3AB", "12345", "GL56 9HQ", "SN7 8RF", "ABCDE",
                     "RG9 1AA", "OX1 2JD", "GL7 1XX", "SN6 8PQ", "CB1 3AA"],
        "monthly_charges": [29.99, "£44.99", 54.99, 64.99, 29.99,
                            44.99, 54.99, "£64.99", 29.99, 44.99],
        "nps_score": [8, 9, 11, 7, None, 6, 10, 8, 9, 7],
        "churn": ["Yes", "no", "Y", "No", "N", "Yes", "No", "1", "No", "0"],
    })

    # --- Dataset 2: completely unrelated (retail sales) — ZERO code changes ---
    sales = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5"],
        "product": ["Widget", "Gadget", "Widget", "Gizmo", "Gadget"],
        "quantity": [3, 1, 5, 2, 4],
        "unit_price": [9.99, 19.99, 9.99, 14.50, 19.99],
        "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
    })

    summarise("Broadband customers", broadband)
    summarise("Retail sales (unrelated)", sales)

    print("\n" + "=" * 60)
    print("PROOF: identical code profiled both datasets with zero changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()