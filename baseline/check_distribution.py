"""
check_distribution.py — Print label distribution and matrix size stats of dataset.h5.

Usage:
    python check_distribution.py
    DATA_DIR=./data python check_distribution.py
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mm_model import MM_SOLVER_NAMES

DATA_DIR = os.getenv("DATA_DIR", "./data")

PAPER_COUNTS = {
    "fbcgsr+jacobi": 2173, "bcgsl+none": 2054, "symmlq+icc": 1201,
    "symmlq+jacobi":  923, "dgmres+none":  650, "gmres+gamg":  640,
    "cr+eisenstat":   598, "symmlq+sor":   582, "fbcgsr+ilu":  562,
    "minres+gamg":    524, "fcg+gamg":     342, "cr+jacobi":   310,
    "cg+ilu":         275, "fgmres+gamg":  226, "cg+eisenstat": 224,
    "cg+bjacobi":     193, "cr+ilu":        68, "cgs+gamg":     49,
    "bcgsl+asm":       29,
}
PAPER_TOTAL = sum(PAPER_COUNTS.values())

with h5py.File(os.path.join(DATA_DIR, "dataset.h5"), "r") as f:
    labels   = f["labels"][:]
    sizes    = f["features"][:, 0].astype(int)  # feature 0 = row_num
    total    = len(labels)

counts = np.bincount(labels, minlength=len(MM_SOLVER_NAMES))
BAR_W  = 25

# ── Distribution table ────────────────────────────────────────────────────────

print(f"\n── Dataset distribution  ({DATA_DIR}/dataset.h5) ──────────────────")
print(f"  Total samples : {total}  |  Paper total : {PAPER_TOTAL:,}\n")
print(f"  {'Solver':<20}  {'Ours':>6}  {'%':>5}  {'Paper':>6}  {'%':>5}  {'Bar (ours)'}")
print(f"  {'-'*20}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*BAR_W}")

for name, cnt in zip(MM_SOLVER_NAMES, counts):
    paper = PAPER_COUNTS.get(name, 0)
    pct       = cnt   / total        * 100
    paper_pct = paper / PAPER_TOTAL  * 100
    bar  = "█" * int(pct / 100 * BAR_W)
    flag = "  ⚠ low" if cnt < 50 else ""
    print(f"  {name:<20}  {cnt:>6}  {pct:>4.1f}%  {paper:>6}  {paper_pct:>4.1f}%  {bar}{flag}")

print(f"\n  {'TOTAL':<20}  {total:>6}  {'100%':>5}  {PAPER_TOTAL:>6}  {'100%':>5}")
empty = sum(1 for c in counts if c == 0)
low   = sum(1 for c in counts if 0 < c < 50)
print(f"\n  Empty classes : {empty}   Low (<50) classes : {low}")

# ── Matrix size table ─────────────────────────────────────────────────────────

print(f"\n── Matrix sizes (n = number of rows) ───────────────────────────────")
print(f"  Overall  min={sizes.min()}  median={int(np.median(sizes))}  "
      f"mean={int(sizes.mean())}  max={sizes.max()}\n")
print(f"  {'Solver':<20}  {'N':>6}  {'min n':>7}  {'med n':>7}  {'mean n':>7}  {'max n':>7}")
print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

for i, (name, cnt) in enumerate(zip(MM_SOLVER_NAMES, counts)):
    if cnt == 0:
        print(f"  {name:<20}  {cnt:>6}  {'—':>7}  {'—':>7}  {'—':>7}  {'—':>7}")
        continue
    s = sizes[labels == i]
    flag = "  ⚠ small" if int(np.median(s)) < 500 else ""
    print(f"  {name:<20}  {cnt:>6}  {s.min():>7}  {int(np.median(s)):>7}  "
          f"{int(s.mean()):>7}  {s.max():>7}{flag}")

print()
