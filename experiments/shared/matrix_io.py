"""
matrix_io.py — Shared matrix loading and classification utilities.

Used by both thesis_experiments and mm_baseline so neither depends on the other.
"""

import logging
import os

import numpy as np
import scipy.io
import scipy.sparse as sp

log = logging.getLogger(__name__)


def _to_csr(raw) -> "sp.csr_matrix | None":
    """Convert a raw sparse or dense array to float64 CSR, or None if complex."""
    if np.iscomplexobj(raw.data if sp.issparse(raw) else raw):
        return None
    return sp.csr_matrix(raw, dtype=np.float64)


def load_matrix(
    path: str,
    require_nonzero_diag: bool = True,
    min_n: int = 100,
    max_n: int = 50000,
) -> "sp.csr_matrix | None":
    """
    Load a .mtx (Matrix Market) or .mat (MATLAB) file and return a real square
    CSR matrix, or None if the matrix is complex, rectangular, empty-row, or
    outside [min_n, max_n].

    require_nonzero_diag=False relaxes the zero-diagonal check (useful for
    curated SuiteSparse lists where only some preconditioners apply).
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".mat":
            data = scipy.io.loadmat(path)
            if "Problem" in data:
                raw = data["Problem"]["A"][0, 0]
            else:
                raw = next(
                    (v for v in data.values()
                     if isinstance(v, np.ndarray) and v.ndim == 2),
                    None,
                )
                if raw is None:
                    log.warning("Cannot find matrix in %s", path)
                    return None
        else:
            raw = scipy.io.mmread(path)
    except Exception as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return None

    A = _to_csr(raw)
    if A is None:
        log.info("Skipping %s — complex matrix.", path)
        return None

    if A.shape[0] != A.shape[1]:
        log.info("Skipping %s — rectangular (%d × %d).", path, *A.shape)
        return None

    n = A.shape[0]
    if not (min_n <= n <= max_n):
        log.info("Skipping %s — size %d outside [%d, %d].", path, n, min_n, max_n)
        return None

    A.eliminate_zeros()
    A.sum_duplicates()
    A.sort_indices()

    if np.any(np.diff(A.indptr) == 0):
        log.info("Skipping %s — has empty rows.", path)
        return None

    if require_nonzero_diag and np.any(np.abs(A.diagonal()) < 1e-300):
        log.info("Skipping %s — has zero diagonal entries.", path)
        return None

    return A


# Alias kept for any existing callers
load_mtx = load_matrix


def classify(A: sp.csr_matrix, isspd: bool = False, issym: bool = False) -> str:
    """
    Return "spd", "sym", or "nonsym" for use with an APPLICABLE solver map.

    isspd / issym come from ssgetpy metadata when available; otherwise
    symmetry is estimated from the matrix itself (||A-Aᵀ||_F / ||A||_F < 0.01).
    """
    if isspd:
        return "spd"
    if issym:
        return "sym"
    frob = sp.linalg.norm(A, "fro")
    if frob < 1e-14:
        return "nonsym"
    if sp.linalg.norm(A - A.T, "fro") / frob < 0.01:
        return "sym"
    return "nonsym"
