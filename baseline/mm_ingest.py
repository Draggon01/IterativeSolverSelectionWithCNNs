"""
mm_ingest.py — Download SuiteSparse matrices and append them to the baseline dataset.

Mirrors src/ingest_suitesparse.py but extracts MM-AutoSolver features
(17 numerical features, 128×128 density image) instead of the main
pipeline's format. Appends to the same dataset.h5 produced by mm_generate.py
so training uses both synthetic and real matrices.

Supports two modes (set via MODE env var):
  auto   (default) — query SuiteSparse via ssgetpy, download automatically
  manual           — scan a local directory (MTX_DIR) for .mtx / .mat files

Environment variables:
  MODE        "auto" or "manual"               (default auto)
  DATA_DIR    Path containing dataset.h5        (default ./data)
  CACHE_DIR   Local cache for downloads         (default ./data/suitesparse_cache)
  MTX_DIR     Directory of .mtx files [manual]  (default ./data/mtx)
  MIN_N       Minimum matrix dimension           (default 100)
  MAX_N       Maximum matrix dimension           (default 50000)
  N_MATRICES  Max SuiteSparse matrices to add    (default 1000)
  ONLY_SPD    Restrict to SPD matrices           (default 0 = all real square)
  SEED        RNG seed for RHS vectors           (default 0)
  MAX_ITER    Max KSP iterations per solver      (default 2000)
  TOL         Convergence tolerance              (default 1e-8)
"""

import glob
import logging
import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Reuse matrix loading and classification from the main pipeline
from ingest_suitesparse import load_matrix, classify, run_ksp

from mm_model import (
    mm_features, mm_density_image, MM_N_FEATURES, MM_IMAGE_SIZE,
    MM_SOLVER_PAIRS, MM_SOLVER_NAMES, MM_SOLVER_IDX, MM_N_SOLVERS, MM_APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODE       = os.getenv("MODE",       "auto")
DATA_DIR   = os.getenv("DATA_DIR",   "./data")
CACHE_DIR  = os.getenv("CACHE_DIR",  os.path.join(DATA_DIR, "suitesparse_cache"))
MTX_DIR    = os.getenv("MTX_DIR",    os.path.join(DATA_DIR, "mtx"))
MIN_N      = int(os.getenv("MIN_N",       "100"))
MAX_N      = int(os.getenv("MAX_N",       "50000"))
N_MATRICES = int(os.getenv("N_MATRICES",  "1000"))
ONLY_SPD   = os.getenv("ONLY_SPD",   "0") == "1"
SEED       = int(os.getenv("SEED",        "0"))
MAX_ITER   = int(os.getenv("MAX_ITER",    "2000"))
TOL        = float(os.getenv("TOL",       "1e-8"))


# ── solver benchmarking ───────────────────────────────────────────────────────

def benchmark(A: sp.csr_matrix, b: np.ndarray, mat_type: str) -> "tuple[int | None, np.ndarray]":
    """Run the paper's 19 solver pairs; return (best_label, runtimes)."""
    all_times  = np.full(MM_N_SOLVERS, np.nan, dtype=np.float32)
    converged: dict[tuple, float] = {}

    for pair in MM_APPLICABLE[mat_type]:
        ksp_type, pc_type = pair
        ok, iters, t = run_ksp(A, b, ksp_type, pc_type)
        if ok:
            converged[pair] = t
            all_times[MM_SOLVER_IDX[pair]] = float(t)
        log.debug("  %-8s+%-8s  ok=%-5s  iters=%-4d  t=%.4fs",
                  ksp_type, pc_type, ok, iters, t)

    if not converged:
        return None, all_times

    best_pair = min(converged, key=converged.__getitem__)
    return int(MM_SOLVER_IDX[best_pair]), all_times


# ── HDF5 helpers ──────────────────────────────────────────────────────────────

def open_or_create_dataset(path: str) -> h5py.File:
    """Open dataset.h5 in append mode, creating all required datasets if absent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, "a")

    def _ensure(name, **kwargs):
        if name not in f:
            f.create_dataset(name, **kwargs)

    _ensure("images",   shape=(0, MM_IMAGE_SIZE, MM_IMAGE_SIZE),
            maxshape=(None, MM_IMAGE_SIZE, MM_IMAGE_SIZE),
            dtype="f4", chunks=(64, MM_IMAGE_SIZE, MM_IMAGE_SIZE))
    _ensure("features", shape=(0, MM_N_FEATURES), maxshape=(None, MM_N_FEATURES),
            dtype="f4", chunks=(256, MM_N_FEATURES))
    _ensure("labels",   shape=(0,), maxshape=(None,), dtype="i4", chunks=(256,))
    _ensure("runtimes", shape=(0, MM_N_SOLVERS), maxshape=(None, MM_N_SOLVERS),
            dtype="f4", chunks=(256, MM_N_SOLVERS))
    _ensure("source",   shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(), chunks=(256,))

    if "solvers"    not in f.attrs: f.attrs["solvers"]    = MM_SOLVER_NAMES
    if "n_features" not in f.attrs: f.attrs["n_features"] = MM_N_FEATURES
    if "image_size" not in f.attrs: f.attrs["image_size"] = MM_IMAGE_SIZE

    return f


def already_ingested(f: h5py.File) -> set[str]:
    if "source" not in f or len(f["source"]) == 0:
        return set()
    return {s.decode() if isinstance(s, bytes) else s for s in f["source"][:]}


def append_sample(f: h5py.File, A: sp.csr_matrix, label: int,
                  times: np.ndarray, source: str) -> None:
    n = len(f["labels"])
    for ds in ("images", "features", "labels", "runtimes", "source"):
        f[ds].resize(n + 1, axis=0)
    f["images"][n]   = mm_density_image(A)
    f["features"][n] = mm_features(A)
    f["labels"][n]   = label
    f["runtimes"][n] = times
    f["source"][n]   = source


# ── ingestion pipeline ────────────────────────────────────────────────────────

def ingest_matrix(f: h5py.File, A: sp.csr_matrix, source: str,
                  isspd: bool, issym: bool, rng: np.random.Generator) -> bool:
    mat_type = classify(A, isspd, issym)
    b        = rng.standard_normal(A.shape[0])

    log.info("  Benchmarking %s  n=%d  nnz=%d  type=%s",
             source, A.shape[0], A.nnz, mat_type)

    label, times = benchmark(A, b, mat_type)
    if label is None:
        log.warning("  No solver converged for %s — skipping.", source)
        return False

    append_sample(f, A, label, times, source)
    f.flush()  # write to disk immediately so a crash doesn't lose data
    log.info("  Saved  best=%s", MM_SOLVER_NAMES[label])
    return True


# ── auto mode (ssgetpy) ───────────────────────────────────────────────────────

def run_auto(f: h5py.File, rng: np.random.Generator) -> None:
    try:
        import ssgetpy
    except ImportError:
        log.error("ssgetpy not installed. Run: pip install ssgetpy")
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

    # Filter by size and real type locally (ssgetpy no longer supports nrows in search)
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

        try:
            matrix.download(destpath=CACHE_DIR, format="MM", extract=True)
        except Exception as exc:
            log.warning("Download failed for %s: %s", source, exc)
            skipped += 1
            continue

        hits = glob.glob(os.path.join(CACHE_DIR, matrix.name, "*.mtx"))
        if not hits:
            log.warning("No .mtx found for %s", source)
            skipped += 1
            continue

        A = load_matrix(hits[0])
        if A is None:
            skipped += 1
            continue

        issym = bool(matrix.isspd) or (getattr(matrix, 'psym', 0) == 1 and getattr(matrix, 'nsym', 0) == 1)
        ok = ingest_matrix(f, A, source, isspd=bool(matrix.isspd),
                           issym=issym, rng=rng)
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

        rel    = os.path.relpath(mtx_path, MTX_DIR)
        source = "manual/" + rel.replace(os.sep, "/").removesuffix(".mtx")

        if source in done:
            log.info("Already ingested %s — skipping.", source)
            saved += 1
            continue

        A = load_matrix(mtx_path)
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

    from petsc4py import PETSc
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
        log.info("Done. Added %d SuiteSparse samples (total=%d).",
                 n_after - n_before, n_after)


if __name__ == "__main__":
    main()
