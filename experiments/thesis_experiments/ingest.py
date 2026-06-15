"""
ingest_suitesparse.py — Download SuiteSparse matrices and append them to dataset.h5.

Supports two modes set via the MODE environment variable:

  auto   (default) — query the SuiteSparse collection via ssgetpy, download
                     matching matrices automatically, and ingest them.

  manual           — scan a local directory (MTX_DIR) for .mtx files and
                     ingest them directly without any download.  Use this if
                     you downloaded matrices by hand from
                     https://sparse.tamu.edu/ or have your own .mtx files.

The script appends to $DATA_DIR/dataset.h5 (created by generate_data.py).
If the file does not exist yet it is created from scratch.  Already-ingested
matrices are skipped so the script is safe to re-run.

Environment variables:
  MODE          "auto" or "manual"               (default auto)
  DATA_DIR      Path containing dataset.h5        (default /workspace/data)
  CACHE_DIR     Local cache for downloaded files  (default /workspace/data/suitesparse_cache)
  MTX_DIR       Directory of .mtx files [manual]  (default /workspace/data/mtx)
  MIN_N         Minimum matrix dimension           (default 100)
  MAX_N         Maximum matrix dimension           (default 50000)
  N_MATRICES    Max matrices to ingest             (default 200)
  ONLY_SPD      Restrict to SPD matrices           (default 0 = all real square)
  MAX_ITER      Hard iteration cap per KSP solver  (default 2000)
  TOL           Convergence tolerance              (default 1e-8)
  SEED          NumPy RNG seed for RHS vectors     (default 0)
"""

import os
import glob
import logging

import numpy as np
import h5py
from petsc4py import PETSc

from generators import run_ksp
from matrix_io import load_matrix, load_mtx, classify
from model import (
    matrix_features, sparsity_image,
    SOLVER_PAIRS, SOLVER_NAMES, SOLVER_IDX, N_SOLVERS, N_FEATURES, IMAGE_SIZE, IMAGE_MODE,
    APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
MODE       = os.getenv("MODE",       "auto")
DATA_DIR   = os.getenv("DATA_DIR",   "/workspace/data")
CACHE_DIR  = os.getenv("CACHE_DIR",  os.path.join(DATA_DIR, "suitesparse_cache"))
MTX_DIR    = os.getenv("MTX_DIR",    os.path.join(DATA_DIR, "mtx"))
MIN_N      = int(os.getenv("MIN_N",       "100"))
MAX_N      = int(os.getenv("MAX_N",       "50000"))
N_MATRICES = int(os.getenv("N_MATRICES",  "200"))
ONLY_SPD   = os.getenv("ONLY_SPD",   "0") == "1"
SEED       = int(os.getenv("SEED",        "0"))


def benchmark(
    A: sp.csr_matrix,
    b: np.ndarray,
    mat_type: str,
) -> tuple[int | None, np.ndarray, np.ndarray]:
    """
    Return (best_label, runtimes, top3) — same contract as generate_data.py.
      runtimes — float32 (N_SOLVERS,), NaN where not converged
      top3     — int8 (3,), indices ranked by wall time, -1 if fewer than k converged
    """
    all_times = np.full(N_SOLVERS, np.nan, dtype=np.float32)
    converged: dict[tuple, float] = {}
    for pair in APPLICABLE[mat_type]:
        ksp_type, pc_type = pair
        ok, iters, t = run_ksp(A, b, ksp_type, pc_type)
        if ok:
            converged[pair] = t
            all_times[SOLVER_IDX[pair]] = float(t)
        log.debug("  %-8s+%-8s  ok=%-5s  iters=%-4d  t=%.4fs",
                  ksp_type, pc_type, ok, iters, t)

    top3 = np.full(3, -1, dtype=np.int8)
    if not converged:
        return None, all_times, top3

    ranked = sorted(converged.items(), key=lambda x: x[1])
    for i, (pair, _) in enumerate(ranked[:3]):
        top3[i] = SOLVER_IDX[pair]

    return int(top3[0]), all_times, top3


# ── HDF5 helpers ──────────────────────────────────────────────────────────────

def open_or_create_dataset(path: str) -> h5py.File:
    """
    Open dataset.h5 in append mode, creating all required datasets if absent.
    If a 'source' dataset is missing from an existing file (created by the old
    generate_data.py), it is added and backfilled with "synthetic".
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, "a")

    def _ensure(name, **kwargs):
        if name not in f:
            f.create_dataset(name, **kwargs)

    _ensure("images",   shape=(0, IMAGE_SIZE, IMAGE_SIZE),
            maxshape=(None, IMAGE_SIZE, IMAGE_SIZE),
            dtype="f4", chunks=(64, IMAGE_SIZE, IMAGE_SIZE))
    _ensure("features", shape=(0, N_FEATURES), maxshape=(None, N_FEATURES),
            dtype="f4", chunks=(256, N_FEATURES))
    _ensure("labels",   shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(256,))
    _ensure("runtimes", shape=(0, N_SOLVERS), maxshape=(None, N_SOLVERS),
            dtype="f4", chunks=(256, N_SOLVERS))
    _ensure("source",     shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(), chunks=(256,))
    _ensure("top3_labels", shape=(0, 3), maxshape=(None, 3),
            dtype="i1", chunks=(256, 3))

    if "solvers" not in f.attrs:
        f.attrs["solvers"] = SOLVER_NAMES
    if "image_mode" not in f.attrs:
        f.attrs["image_mode"] = IMAGE_MODE

    # Backfill 'source' for rows written by old generate_data.py (no source field)
    n_rows = len(f["labels"])
    n_src  = len(f["source"])
    if n_src < n_rows:
        f["source"].resize(n_rows, axis=0)
        f["source"][n_src:n_rows] = "synthetic"
        log.info("Backfilled %d rows with source='synthetic'.", n_rows - n_src)

    return f


def already_ingested(f: h5py.File) -> set[str]:
    """Return the set of source strings already present in the dataset."""
    if "source" not in f or len(f["source"]) == 0:
        return set()
    return set(s.decode() if isinstance(s, bytes) else s for s in f["source"][:])


def append_sample(
    f:      h5py.File,
    A:      sp.csr_matrix,
    label:  int,
    times:  np.ndarray,
    top3:   np.ndarray,
    source: str,
) -> None:
    n = len(f["labels"])
    for ds in ("images", "features", "labels", "runtimes", "source", "top3_labels"):
        f[ds].resize(n + 1, axis=0)
    f["images"][n]      = sparsity_image(A)
    f["features"][n]    = matrix_features(A)
    f["labels"][n]      = label
    f["runtimes"][n]    = times
    f["source"][n]      = source
    f["top3_labels"][n] = top3


# ── ingestion pipeline ────────────────────────────────────────────────────────

def ingest_matrix(
    f:       h5py.File,
    A:       sp.csr_matrix,
    source:  str,
    isspd:   bool,
    issym:   bool,
    rng:     np.random.Generator,
) -> bool:
    """
    Benchmark A, extract features, and append to the dataset.
    Returns True if the sample was saved, False if it was skipped.
    """
    mat_type = classify(A, isspd, issym)
    b        = rng.standard_normal(A.shape[0])

    log.info("  Benchmarking %s  n=%d  nnz=%d  type=%s",
             source, A.shape[0], A.nnz, mat_type)

    label, times, top3 = benchmark(A, b, mat_type)
    if label is None:
        log.warning("  No solver converged for %s — skipping.", source)
        return False

    append_sample(f, A, label, times, top3, source)
    log.info("  Saved  best=%s  times_ms=%s",
             SOLVER_NAMES[label],
             {SOLVER_NAMES[i]: f"{times[i]*1000:.1f}" for i in range(N_SOLVERS)
              if not np.isnan(times[i])})
    return True


# ── auto mode (ssgetpy) ───────────────────────────────────────────────────────

def run_auto(f: h5py.File, rng: np.random.Generator) -> None:
    try:
        import ssgetpy
    except ImportError:
        log.error(
            "ssgetpy is not installed.  Run: pip install ssgetpy\n"
            "Or switch to manual mode: MODE=manual MTX_DIR=/path/to/mtx"
        )
        return

    done    = already_ingested(f)
    saved   = skipped = 0

    search_kwargs: dict = dict(limit=N_MATRICES * 5)
    if ONLY_SPD:
        search_kwargs["isspd"] = True

    log.info("Querying SuiteSparse collection ...")
    try:
        results = ssgetpy.search(**search_kwargs)
    except Exception as exc:
        log.error("SuiteSparse query failed: %s", exc)
        return

    results = [m for m in results
               if MIN_N <= m.rows <= MAX_N
               and getattr(m, 'dtype', 'real') == 'real']
    log.info("Found %d candidate matrices after filtering (n=[%d,%d]).",
             len(results), MIN_N, MAX_N)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for matrix in results:
        if saved >= N_MATRICES:
            break

        source = f"suitesparse/{matrix.group}/{matrix.name}"

        if source in done:
            log.info("Already ingested %s — skipping.", source)
            saved += 1
            continue

        # Download (cached — ssgetpy skips if the file already exists)
        try:
            matrix.download(destpath=CACHE_DIR, format="MM", extract=True)
        except Exception as exc:
            log.warning("Download failed for %s: %s", source, exc)
            skipped += 1
            continue

        # Locate the .mtx file
        pattern = os.path.join(CACHE_DIR, matrix.name, "*.mtx")
        hits    = glob.glob(pattern)
        if not hits:
            log.warning("No .mtx found at %s", pattern)
            skipped += 1
            continue
        mtx_path = hits[0]

        A = load_matrix(mtx_path, min_n=MIN_N, max_n=MAX_N)
        if A is None:
            skipped += 1
            continue

        issym = bool(matrix.isspd) or (getattr(matrix, 'psym', 0) == 1 and getattr(matrix, 'nsym', 0) == 1)
        ok = ingest_matrix(f, A, source,
                           isspd=bool(matrix.isspd),
                           issym=issym,
                           rng=rng)
        if ok:
            saved += 1
        else:
            skipped += 1

    log.info("Auto mode complete: saved=%d  skipped=%d", saved, skipped)


# ── manual mode ───────────────────────────────────────────────────────────────

def run_manual(f: h5py.File, rng: np.random.Generator) -> None:
    mtx_files = sorted(
        glob.glob(os.path.join(MTX_DIR, "**", "*.mtx"), recursive=True) +
        glob.glob(os.path.join(MTX_DIR, "**", "*.mat"), recursive=True)
    )
    if not mtx_files:
        log.error("No .mtx or .mat files found under %s", MTX_DIR)
        return

    done    = already_ingested(f)
    saved   = skipped = 0

    for mtx_path in mtx_files:
        if saved >= N_MATRICES:
            break

        # Derive a stable source identifier from the file path
        rel    = os.path.relpath(mtx_path, MTX_DIR)
        source = "manual/" + rel.replace(os.sep, "/").removesuffix(".mtx")

        if source in done:
            log.info("Already ingested %s — skipping.", source)
            saved += 1
            continue

        log.info("Loading %s", mtx_path)
        A = load_matrix(mtx_path, min_n=MIN_N, max_n=MAX_N)
        if A is None:
            skipped += 1
            continue

        ok = ingest_matrix(f, A, source, isspd=False, issym=False, rng=rng)
        if ok:
            saved += 1
        else:
            skipped += 1

    log.info("Manual mode complete: saved=%d  skipped=%d", saved, skipped)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    PETSc.Options()["ksp_error_if_not_converged"] = False

    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    rng     = np.random.default_rng(SEED)

    with open_or_create_dataset(h5_path) as f:
        n_before = len(f["labels"])
        log.info("Dataset at %s — %d existing samples.", h5_path, n_before)

        if MODE == "manual":
            log.info("Manual mode — scanning %s", MTX_DIR)
            run_manual(f, rng)
        else:
            log.info("Auto mode — querying SuiteSparse (n=[%d,%d]  max=%d)",
                     MIN_N, MAX_N, N_MATRICES)
            run_auto(f, rng)

        n_after = len(f["labels"])
        log.info("Done. Added %d samples (total=%d).", n_after - n_before, n_after)


if __name__ == "__main__":
    main()
