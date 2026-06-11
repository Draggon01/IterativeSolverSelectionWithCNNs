"""
mm_trim.py — Cap all over-represented solver classes in dataset.h5 to MAX_PER_CLASS.

By default every class that exceeds MAX_PER_CLASS is trimmed down randomly.
Use CAP to restrict trimming to a specific comma-separated list of solver names.

Usage:
    python mm_trim.py                              # trim all classes > 2500
    MAX_PER_CLASS=1500 python mm_trim.py           # different cap
    CAP=fbcgsr+jacobi,cg+eisenstat python mm_trim.py  # only those two classes
    DATA_DIR=./data python mm_trim.py

Writes dataset_trimmed.h5 alongside the original, then renames:
    dataset.h5.bak  ← original
    dataset.h5      ← trimmed
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mm_model import MM_SOLVER_NAMES

DATA_DIR      = os.getenv("DATA_DIR",      "./data")
MAX_PER_CLASS = int(os.getenv("MAX_PER_CLASS", "2500"))
SEED          = int(os.getenv("SEED",          "42"))
_cap_env      = os.getenv("CAP", "").strip()

src_path = os.path.join(DATA_DIR, "dataset.h5")
tmp_path = os.path.join(DATA_DIR, "dataset_trimmed.h5")
bak_path = os.path.join(DATA_DIR, "dataset.h5.bak")

# If CAP is set, only trim those classes; otherwise trim all that exceed the cap
if _cap_env:
    cap_names = [n.strip() for n in _cap_env.split(",") if n.strip()]
    for name in cap_names:
        if name not in MM_SOLVER_NAMES:
            print(f"ERROR: '{name}' not in MM_SOLVER_NAMES. Check spelling.")
            sys.exit(1)
    cap_set = set(cap_names)
    print(f"\nSelective trim — capping to {MAX_PER_CLASS}: {sorted(cap_set)}")
else:
    cap_set = None   # trim all classes that exceed the limit
    print(f"\nTrimming all classes that exceed {MAX_PER_CLASS} entries.")

rng = np.random.default_rng(SEED)

with h5py.File(src_path, "r") as src:
    labels = src["labels"][:]
    n_orig = len(labels)

    keep_indices = []
    for label_id, name in enumerate(MM_SOLVER_NAMES):
        idx = np.where(labels == label_id)[0]
        should_cap = (cap_set is None or name in cap_set) and len(idx) > MAX_PER_CLASS
        if should_cap:
            chosen = rng.choice(idx, size=MAX_PER_CLASS, replace=False)
            print(f"  {name:<22}: {len(idx):>5} → {MAX_PER_CLASS}")
            keep_indices.append(chosen)
        else:
            keep_indices.append(idx)

    keep   = np.sort(np.concatenate(keep_indices))
    n_keep = len(keep)

    print(f"\n  Original samples : {n_orig}")
    print(f"  Kept samples     : {n_keep}  (removed {n_orig - n_keep})")

    with h5py.File(tmp_path, "w") as dst:
        for key in ("images", "features", "labels", "runtimes", "source"):
            data   = src[key][keep]
            src_ds = src[key]
            maxshape = tuple(None if i == 0 else s for i, s in enumerate(data.shape))
            kwargs = dict(maxshape=maxshape, chunks=src_ds.chunks or True)
            if src_ds.dtype.kind == "O":
                kwargs["dtype"] = h5py.string_dtype()
            dst.create_dataset(key, data=data, **kwargs)
        for attr_key, attr_val in src.attrs.items():
            dst.attrs[attr_key] = attr_val

os.rename(src_path, bak_path)
os.rename(tmp_path, src_path)

print(f"\n  Original backed up → {bak_path}")
print(f"  Trimmed dataset    → {src_path}")

with h5py.File(src_path, "r") as f:
    new_labels = f["labels"][:]

counts = np.bincount(new_labels, minlength=len(MM_SOLVER_NAMES))
print(f"\n  New distribution ({n_keep} total):")
for name, cnt in zip(MM_SOLVER_NAMES, counts):
    bar = "█" * int(cnt / n_keep * 40) if n_keep else ""
    print(f"    {name:<22}  {cnt:>5}  {cnt/n_keep*100:>4.1f}%  {bar}")
print()
