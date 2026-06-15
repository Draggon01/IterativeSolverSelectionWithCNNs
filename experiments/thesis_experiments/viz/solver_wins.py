"""
solver_wins.py — Plot and print the solver win distribution from dataset.h5.

Saves solver_wins.png alongside this script and prints a text summary to stdout.
Runs headlessly — no display required.

Environment variables:
  DATA_DIR   Directory containing dataset.h5  (default /workspace/data)
  OUT_DIR    Where to write solver_wins.png   (default same directory as this script)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

DATA_DIR = os.getenv("DATA_DIR", "/workspace/data")
OUT_DIR  = os.getenv("OUT_DIR",  os.path.dirname(os.path.abspath(__file__)))

h5_path  = os.path.join(DATA_DIR, "dataset.h5")

if not os.path.exists(h5_path):
    print(f"ERROR: dataset not found at {h5_path}", file=sys.stderr)
    sys.exit(1)

with h5py.File(h5_path, "r") as f:
    labels  = f["labels"][:]
    solvers = [s.decode() if isinstance(s, bytes) else s for s in f.attrs["solvers"]]

n_solvers = len(solvers)
counts    = np.bincount(labels, minlength=n_solvers)
total     = len(labels)

# Sort by count descending for readability
order   = np.argsort(counts)[::-1]
names   = [solvers[i] for i in order]
vals    = counts[order]
pcts    = vals / total * 100

# ── text summary ──────────────────────────────────────────────────────────────
print(f"\nSolver win distribution  ({total} samples, {DATA_DIR})")
print(f"{'Rank':<5} {'Solver':<25} {'Wins':>6}  {'%':>6}")
print("─" * 46)
for rank, (name, cnt, pct) in enumerate(zip(names, vals, pcts), 1):
    bar = "█" * int(pct / 2)
    print(f"  {rank:<3} {name:<25} {cnt:>6}  {pct:>5.1f}%  {bar}")
print()

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(5, n_solvers * 0.42)))

cmap   = plt.cm.tab20(np.linspace(0, 1, n_solvers))
bars   = ax.barh(names[::-1], vals[::-1], color=cmap[::-1], edgecolor="white", linewidth=0.5)

for bar, cnt, pct in zip(bars, vals[::-1], pcts[::-1]):
    ax.text(
        bar.get_width() + total * 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{cnt:,}  ({pct:.1f}%)",
        va="center", fontsize=8.5,
    )

ax.set_xlabel("Number of wins (samples where this solver was fastest)")
ax.set_title(
    f"Solver Win Distribution\n{total:,} samples · {DATA_DIR}",
    fontsize=12, fontweight="bold",
)
ax.set_xlim(0, vals.max() * 1.22)
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solver_wins.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")
