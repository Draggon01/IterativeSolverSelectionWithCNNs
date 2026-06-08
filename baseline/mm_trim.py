"""
mm_trim.py — Cap specific solver classes in dataset.h5 to at most MAX_PER_CLASS entries.

Usage:
    python mm_trim.py                                   # caps defaults to 2500
    CAP=fbcgsr+jacobi,cg+eisenstat MAX_PER_CLASS=2500 python mm_trim.py
    DATA_DIR=./data python mm_trim.py

Writes a new file dataset_trimmed.h5 alongside the original, then renames:
    dataset.h5.bak  ← original
    dataset.h5      ← trimmed
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mm_model import MM_SOLVER_NAMES

DATA_DIR      = os.getenv("DATA_DIR", "./data")
CAP           = os.getenv("CAP", "fbcgsr+jacobi,cg+eisenstat").split(",")
MAX_PER_CLASS = int(os.getenv("MAX_PER_CLASS", "2500"))
SEED          = int(os.getenv("SEED", "42"))

src_path = os.path.join(DATA_DIR, "dataset.h5")
tmp_path = os.path.join(DATA_DIR, "dataset_trimmed.h5")
bak_path = os.path.join(DATA_DIR, "dataset.h5.bak")

cap_ids = {}
for name in CAP:
    name = name.strip()
    if name not in MM_SOLVER_NAMES:
        print(f"ERROR: '{name}' not in MM_SOLVER_NAMES. Check spelling.")
        sys.exit(1)
    cap_ids[MM_SOLVER_NAMES.index(name)] = name

print(f"\nCapping classes to {MAX_PER_CLASS} entries: {list(cap_ids.values())}")

rng = np.random.default_rng(SEED)

with h5py.File(src_path, "r") as src:
    labels = src["labels"][:]
    n_orig = len(labels)

    keep_indices = []
    for label_id in range(len(MM_SOLVER_NAMES)):
        idx = np.where(labels == label_id)[0]
        if label_id in cap_ids and len(idx) > MAX_PER_CLASS:
            idx = rng.choice(idx, size=MAX_PER_CLASS, replace=False)
            print(f"  {MM_SOLVER_NAMES[label_id]:<20}: {len(np.where(labels == label_id)[0]):>5} → {MAX_PER_CLASS}")
        keep_indices.append(idx)

    keep = np.sort(np.concatenate(keep_indices))
    n_keep = len(keep)

    print(f"\n  Original samples : {n_orig}")
    print(f"  Kept samples     : {n_keep}  (removed {n_orig - n_keep})")

    with h5py.File(tmp_path, "w") as dst:
        for key in ("images", "features", "labels", "runtimes", "source"):
            data = src[key][keep]
            src_ds = src[key]
            maxshape = tuple(None if i == 0 else s for i, s in enumerate(data.shape))
            kwargs = dict(maxshape=maxshape, chunks=src_ds.chunks or True)
            if src_ds.dtype.kind == 'O':  # variable-length string
                kwargs["dtype"] = h5py.string_dtype()
            dst.create_dataset(key, data=data, **kwargs)
        for attr_key, attr_val in src.attrs.items():
            dst.attrs[attr_key] = attr_val

# Swap files
os.rename(src_path, bak_path)
os.rename(tmp_path, src_path)

print(f"\n  Original backed up → {bak_path}")
print(f"  Trimmed dataset    → {src_path}")

# Print new distribution
with h5py.File(src_path, "r") as f:
    new_labels = f["labels"][:]

counts = np.bincount(new_labels, minlength=len(MM_SOLVER_NAMES))
print(f"\n  New distribution ({n_keep} total):")
for name, cnt in zip(MM_SOLVER_NAMES, counts):
    if cnt > 0:
        bar = "█" * int(cnt / n_keep * 40)
        print(f"    {name:<20}  {cnt:>5}  {cnt/n_keep*100:>4.1f}%  {bar}")
print()
