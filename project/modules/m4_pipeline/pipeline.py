"""
MODULE 4 — PIPELINE ORCHESTRATION
=================================
Chain Modules 1 -> 2 -> 3 into ONE runnable pipeline.

  python modules/m4_pipeline/pipeline.py --input data/raw/data.csv --output outputs/

WHAT TO BUILD (Week 4 — keep it lightweight)
  - run profiling -> cleaning -> validation in sequence
  - one CLI command, clear console logging of each stage
  - a minimal Dockerfile so "it runs anywhere" (already stubbed below)

Scope note: the brief mentions DVC, Git LFS, an NLP classifier, full CI.
SKIP these solo. "Runs end-to-end with one command" satisfies completeness.
List the rest under "future work" in the README.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODULES = Path(__file__).resolve().parents[1]


def run_stage(label: str, script: Path, input_path: str | None = None) -> None:
    cmd = [sys.executable, str(script)]
    if input_path:
        cmd += ["--input", input_path]
    print(f"\n=== {label} ===")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end data-quality pipeline")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", default="outputs/", help="Output folder")
    args = parser.parse_args()

    run_stage("M1 Profile",  MODULES / "m1_profiling" / "profile.py",  args.input)
    run_stage("M2 Clean",    MODULES / "m2_cleaning"  / "clean.py",    args.input)
    run_stage("M3 Validate", MODULES / "m3_validation" / "validate.py")

    print("\n✅ Pipeline complete. Reports are in outputs/.")


if __name__ == "__main__":
    main()
