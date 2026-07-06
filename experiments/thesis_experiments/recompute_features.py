"""
recompute_features.py — Recompute matrix features in-place in an existing dataset.h5.

Use this after adding new features to model.py's matrix_features() to update
the base dataset without re-running the slow PETSc datagen step.

Reconstructs each matrix from stored CSR data (synthetic rows) or the
SuiteSparse cache (suitesparse/* rows), then overwrites the features dataset.

Environment variables:
  DATA_DIR    Directory containing dataset.h5  (default /workspace/data)
  CACHE_DIR   SuiteSparse .mtx cache           (default /workspace/cache)
  CHUNK       Rows to process per log message  (default 200)
"""

import glob
import logging
import os

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

from model import matrix_features, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR  = os.getenv("DATA_DIR",  "/workspace/data")
CACHE_DIR = os.getenv("CACHE_DIR", "/workspace/cache")
CHUNK     = int(os.getenv("CHUNK", "200"))


def _has_stored_csr(f: h5py.File, i: int) -> bool:
    return (
        "mat_data" in f
        and i < len(f["mat_data"])
        and len(f["mat_data"][i]) > 0
        and len(f["mat_indptr"][i]) > 0
    )


def _load_from_stored(f: h5py.File, i: int) -> sp.csr_matrix:
    data    = f["mat_data"][i].astype(np.float64)
    indices = f["mat_indices"][i].astype(np.int32)
    indptr  = f["mat_indptr"][i].astype(np.int32)
    shape   = tuple(f["mat_shape"][i])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _load_from_suitesparse(source: str) -> sp.csr_matrix:
    import ssgetpy
    _, group, name = source.split("/", 2)
    pattern = os.path.join(CACHE_DIR, name, "*.mtx")
    files   = glob.glob(pattern)
    if not files:
        results = ssgetpy.search(name=name, group=group)
        if not results:
            raise RuntimeError(f"Matrix not found in SuiteSparse: {source}")
        results[0].download(destpath=CACHE_DIR, format="MM", extract=True)
        files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No .mtx after download: {source}")
    exact  = [f for f in files if os.path.splitext(os.path.basename(f))[0] == name]
    chosen = exact[0] if exact else sorted(files)[0]
    return sp.csr_matrix(scipy.io.mmread(chosen))


def reconstruct_matrix(f: h5py.File, i: int) -> sp.csr_matrix | None:
    if _has_stored_csr(f, i):
        return _load_from_stored(f, i)
    raw_source = f["source"][i]
    source = raw_source.decode() if isinstance(raw_source, bytes) else raw_source
    if source.startswith("suitesparse/"):
        try:
            return _load_from_suitesparse(source)
        except Exception as e:
            log.warning("Row %d (%s): %s — skipping (features set to 0)", i, source, e)
            return None
    log.warning("Row %d (source=%r): no stored CSR and no fallback — features set to 0", i, source)
    return None


def main() -> None:
    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        n = len(f["labels"])
        old_n_feat = f["features"].shape[1]

    log.info("Dataset   : %s  (%d samples)", h5_path, n)
    log.info("Old features shape: (%d, %d) → new: (%d, %d)", n, old_n_feat, n, N_FEATURES)

    # Build new features array in memory, then write back atomically
    new_features = np.zeros((n, N_FEATURES), dtype=np.float32)
    skipped = 0

    with h5py.File(h5_path, "r") as f:
        for i in range(n):
            A = reconstruct_matrix(f, i)
            if A is None:
                skipped += 1
            else:
                new_features[i] = matrix_features(A)

            if (i + 1) % CHUNK == 0 or i + 1 == n:
                log.info("  Computed %d / %d  (skipped: %d)", i + 1, n, skipped)

    # Overwrite features dataset in-place (delete + recreate to allow shape change)
    with h5py.File(h5_path, "a") as f:
        del f["features"]
        f.create_dataset("features", data=new_features, dtype="f4",
                         chunks=(min(512, n), N_FEATURES))
        f.attrs["n_features"] = N_FEATURES
    log.info("Done — features updated to shape (%d, %d) in %s", n, N_FEATURES, h5_path)


if __name__ == "__main__":
    main()
