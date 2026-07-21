"""
Generate a matrix-size histogram for the thesis (Chapter 3, Dataset Statistics).
Output: thesis/workdir/figures/dataset_size_histogram.pdf

Run from any directory:
    python experiments/thesis_experiments/plot_size_histogram.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

H5_PATH  = os.path.join(os.path.dirname(__file__), "data/base/dataset.h5")
OUT_PATH = os.path.join(os.path.dirname(__file__), "../../thesis/workdir/figures/dataset_size_histogram.pdf")

with h5py.File(H5_PATH, "r") as f:
    n_rows = f["mat_shape"][:, 0].astype(np.int64)

n_rows = n_rows[n_rows > 0]

fig, ax = plt.subplots(figsize=(6, 3))

bins = np.logspace(np.log10(n_rows.min()), np.log10(n_rows.max()), 40)
ax.hist(n_rows, bins=bins, color="#4878CF", edgecolor="white", linewidth=0.4)

ax.set_xscale("log")
ax.set_xlabel("Matrix size $n$ (number of rows)", fontsize=11)
ax.set_ylabel("Number of matrices", fontsize=11)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
