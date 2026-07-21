"""
meta.py — Print experiment configuration: solver pairs and extracted features.

Intended to be captured into results_summary.txt at the start of a run so
the exact configuration is always recorded alongside the results.
"""

import datetime
import os

from model import SOLVER_NAMES, SOLVER_PAIRS, FEATURE_NAMES, IMAGE_MODE, IMAGE_SIZE, NO_CNN

EXPERIMENT = os.getenv("EXPERIMENT", "default")

print("=" * 60)
print("  Experiment metadata")
print(f"  Experiment : {EXPERIMENT}")
print(f"  Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  IMAGE_MODE : {IMAGE_MODE}")
print(f"  IMAGE_SIZE : {IMAGE_SIZE}")
print(f"  NO_CNN     : {NO_CNN}")
print("=" * 60)

print(f"\nSolver pairs ({len(SOLVER_PAIRS)}):")
for i, (ksp, pc) in enumerate(SOLVER_PAIRS):
    print(f"  {i:>2}  {ksp}+{pc}")

print(f"\nExtracted features ({len(FEATURE_NAMES)}):")
for i, name in enumerate(FEATURE_NAMES):
    print(f"  {i:>2}  {name}")

print()
print("=" * 60)
print("  Results")
print("=" * 60)
print()
