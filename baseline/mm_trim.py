"""
mm_trim.py — Trim the MM-AutoSolver baseline dataset.

Three modes (set via MODE env var):

  max_per_class     (default) — cap each class that exceeds MAX_PER_CLASS
                                samples by random down-sampling

  suitesparse_only  — discard all synthetic samples; keep only real matrices
                      whose source starts with "suitesparse/"

  paper_sizes       — cap each solver class at the paper's original sample
                      count (Table 3, Xiong et al. 2025). Classes below the
                      cap are kept in full; classes above are randomly
                      down-sampled. Add SUITESPARSE_ONLY=1 to also filter
                      out synthetic samples first.

Writes dataset_trimmed.h5 alongside the original, then renames:
    dataset.h5.bak  ← original
    dataset.h5      ← trimmed

Environment variables:
  MODE              max_per_class | suitesparse_only | paper_sizes
  DATA_DIR          Directory containing dataset.h5  (default ./data)
  MAX_PER_CLASS     Cap per class [max_per_class]    (default 2500)
  CAP               Comma-separated solver names to cap [max_per_class]
  SUITESPARSE_ONLY  Pre-filter to SuiteSparse [paper_sizes] (default 0)
  SEED              RNG seed for downsampling         (default 42)
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from mm_model import MM_SOLVER_NAMES

MODE             = os.getenv("MODE",          "max_per_class")
DATA_DIR         = os.getenv("DATA_DIR",      "./data")
MAX_PER_CLASS    = int(os.getenv("MAX_PER_CLASS", "2500"))
SEED             = int(os.getenv("SEED",          "42"))
SUITESPARSE_ONLY = os.getenv("SUITESPARSE_ONLY", "0") == "1"
_cap_env         = os.getenv("CAP", "").strip()

src_path = os.path.join(DATA_DIR, "dataset.h5")
tmp_path = os.path.join(DATA_DIR, "dataset_trimmed.h5")
bak_path = os.path.join(DATA_DIR, "dataset.h5.bak")

# Paper's full dataset counts per class (train + test, Table 3, Xiong et al. 2025).
# The paper reports training counts; full totals derived from the ~90/10 split
# (confirmed: fbcgsr+jacobi train=2173, test=231, total=2404).
PAPER_COUNTS: dict[int, int] = {
    0:  2404,  # fbcgsr+jacobi  (2173 train + 231 test)
    1:  2272,  # bcgsl+none
    2:  1329,  # symmlq+icc
    3:  1021,  # symmlq+jacobi
    4:   719,  # dgmres+none
    5:   708,  # gmres+gamg
    6:   662,  # cr+eisenstat
    7:   644,  # symmlq+sor
    8:   622,  # fbcgsr+ilu
    9:   580,  # minres+gamg
    10:  378,  # fcg+gamg
    11:  343,  # cr+jacobi
    12:  304,  # cg+ilu
    13:  250,  # fgmres+gamg
    14:  248,  # cg+eisenstat
    15:  214,  # cg+bjacobi
    16:   75,  # cr+ilu
    17:   54,  # cgs+gamg
    18:   32,  # bcgsl+asm
}

rng = np.random.default_rng(SEED)


def select_max_per_class(labels: np.ndarray, sources: list[str], cap_set) -> np.ndarray:
    print(f"\nmode=max_per_class  cap={MAX_PER_CLASS}"
          + (f"  classes={sorted(cap_set)}" if cap_set else "  (all classes)")
          + ("  prefer_suitesparse=yes" if SUITESPARSE_ONLY else ""))
    keep = []
    for label_id, name in enumerate(MM_SOLVER_NAMES):
        idx = np.where(labels == label_id)[0]
        should_cap = (cap_set is None or name in cap_set) and len(idx) > MAX_PER_CLASS
        if not should_cap:
            keep.append(idx)
            continue

        if SUITESPARSE_ONLY:
            # Fill cap with SuiteSparse first, top up with synthetic if needed
            real_idx  = np.array([i for i in idx if sources[i].startswith("suitesparse")])
            synth_idx = np.array([i for i in idx if not sources[i].startswith("suitesparse")])
            if len(real_idx) >= MAX_PER_CLASS:
                chosen = rng.choice(real_idx, size=MAX_PER_CLASS, replace=False)
            else:
                n_synth = MAX_PER_CLASS - len(real_idx)
                extra   = rng.choice(synth_idx, size=min(n_synth, len(synth_idx)), replace=False)
                chosen  = np.concatenate([real_idx, extra])
            print(f"  {name:<25}: {len(idx):>6} → {len(chosen)}"
                  f"  (real={len(real_idx)}, synth used={len(chosen)-len(real_idx) if len(real_idx) < MAX_PER_CLASS else 0})")
        else:
            chosen = rng.choice(idx, size=MAX_PER_CLASS, replace=False)
            print(f"  {name:<25}: {len(idx):>6} → {MAX_PER_CLASS}")

        keep.append(chosen)
    return np.sort(np.concatenate(keep))


def select_suitesparse_only(labels: np.ndarray, sources: list[str]) -> np.ndarray:
    print("\nmode=suitesparse_only — removing synthetic samples")
    mask = np.array([s.startswith("suitesparse") for s in sources])
    selected = np.where(mask)[0]
    print(f"  {len(labels)} → {len(selected)} samples")
    return selected


def select_paper_sizes(labels: np.ndarray, sources: list[str]) -> np.ndarray:
    print("\nmode=paper_sizes — capping each class to paper's original counts")
    pool = np.arange(len(labels))

    if SUITESPARSE_ONLY:
        mask = np.array([s.startswith("suitesparse") for s in sources])
        pool = pool[mask]
        print(f"  Pre-filter SuiteSparse: {len(labels)} → {len(pool)} samples")

    keep = []
    for cls_idx, cap in PAPER_COUNTS.items():
        idx = pool[labels[pool] == cls_idx]
        if len(idx) == 0:
            print(f"  {MM_SOLVER_NAMES[cls_idx]:<25}: 0 samples (class empty — skipped)")
            continue
        if len(idx) <= cap:
            chosen = idx
        else:
            chosen = rng.choice(idx, size=cap, replace=False)
        print(f"  {MM_SOLVER_NAMES[cls_idx]:<25}: {len(idx):>6} → {len(chosen)}")
        keep.append(chosen)

    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)


def write_subset(src: h5py.File, keep: np.ndarray) -> None:
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


def main() -> None:
    with h5py.File(src_path, "r") as src:
        labels  = src["labels"][:]
        sources = [s.decode() if isinstance(s, bytes) else s for s in src["source"][:]]
        n_orig  = len(labels)

        if MODE == "suitesparse_only":
            keep = select_suitesparse_only(labels, sources)
        elif MODE == "paper_sizes":
            keep = select_paper_sizes(labels, sources)
        else:  # max_per_class (default)
            if _cap_env:
                cap_set = set(_cap_env.split(","))
                for name in cap_set:
                    if name not in MM_SOLVER_NAMES:
                        print(f"ERROR: '{name}' not in MM_SOLVER_NAMES.")
                        sys.exit(1)
            else:
                cap_set = None
            keep = select_max_per_class(labels, sources, cap_set)

        if len(keep) == 0:
            print("ERROR: No samples selected — output not written.")
            sys.exit(1)

        print(f"\n  Original : {n_orig} samples")
        print(f"  Kept     : {len(keep)} samples  (removed {n_orig - len(keep)})")

        write_subset(src, keep)

    os.rename(src_path, bak_path)
    os.rename(tmp_path, src_path)

    print(f"\n  Original backed up → {bak_path}")
    print(f"  Trimmed dataset    → {src_path}")

    with h5py.File(src_path, "r") as f:
        new_labels = f["labels"][:]

    counts = np.bincount(new_labels, minlength=len(MM_SOLVER_NAMES))
    n_keep = len(new_labels)
    print(f"\n  Distribution ({n_keep} total):")
    for name, cnt in zip(MM_SOLVER_NAMES, counts):
        bar = "█" * int(cnt / n_keep * 40) if n_keep else ""
        print(f"    {name:<25}  {cnt:>5}  {cnt/n_keep*100:>4.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
