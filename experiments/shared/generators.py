"""
generators.py — Shared matrix generators and PETSc KSP runner.

Used by both thesis_experiments and mm_baseline so neither depends on the other.

Environment variables:
  MAX_ITER   Hard iteration cap per KSP solve  (default 2000)
  TOL        Relative residual tolerance       (default 1e-8)
"""

import os
import time
import logging

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

log = logging.getLogger(__name__)

MAX_ITER = int(os.getenv("MAX_ITER", "2000"))
TOL      = float(os.getenv("TOL",    "1e-8"))


# ── random matrix generators ──────────────────────────────────────────────────

def random_spd(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Random symmetric positive-definite matrix (diagonally dominant construction)."""
    A     = sp.random(n, n, density=density, format="csr",
                      random_state=rng, dtype=np.float64)
    A     = A + A.T
    shift = np.array(np.abs(A).sum(axis=1)).ravel() + 1.0
    A     = A + sp.diags(shift, format="csr", dtype=np.float64)
    return A.tocsr()


def random_nonsymmetric(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Random non-symmetric diagonally dominant sparse matrix."""
    A       = sp.random(n, n, density=density, format="csr",
                        random_state=rng, dtype=np.float64)
    row_sum = np.array(np.abs(A).sum(axis=1)).ravel()
    A       = A + sp.diags(row_sum + 1.0, format="csr", dtype=np.float64)
    return A.tocsr()


def poisson_2d(nx: int, ny: int | None = None) -> sp.csr_matrix:
    """5-point finite-difference discretisation of −∇²u on an nx × ny grid."""
    if ny is None:
        ny = nx
    n  = nx * ny
    oh = np.full(n - 1, -1.0)
    oh[nx - 1::nx] = 0.0
    return sp.diags(
        [np.full(n - nx, -1.0), oh, np.full(n, 4.0), oh.copy(), np.full(n - nx, -1.0)],
        [-nx, -1, 0, 1, nx],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def poisson_3d(nx: int, ny: int | None = None, nz: int | None = None) -> sp.csr_matrix:
    """7-point finite-difference discretisation of −∇²u on an nx × ny × nz grid."""
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx
    n   = nx * ny * nz
    nxy = nx * ny
    return sp.diags(
        [
            np.full(n - nxy, -1.0),
            np.full(n - nx,  -1.0),
            np.full(n - 1,   -1.0),
            np.full(n,        6.0),
            np.full(n - 1,   -1.0),
            np.full(n - nx,  -1.0),
            np.full(n - nxy, -1.0),
        ],
        [-nxy, -nx, -1, 0, 1, nx, nxy],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def sample_matrix(rng: np.random.Generator) -> tuple[sp.csr_matrix, str]:
    """Uniformly pick a matrix type and random size; return (A, type_name)."""
    choice = int(rng.integers(0, 4))
    if choice == 0:
        n = int(rng.integers(100, 500))
        d = float(rng.uniform(0.02, 0.08))
        return random_spd(n, d, rng), "spd"
    elif choice == 1:
        n = int(rng.integers(100, 500))
        d = float(rng.uniform(0.02, 0.08))
        return random_nonsymmetric(n, d, rng), "nonsym"
    elif choice == 2:
        nx = int(rng.integers(10, 50))
        return poisson_2d(nx), "poisson2d"
    else:
        nx = int(rng.integers(5, 20))
        return poisson_3d(nx), "poisson3d"


# ── PETSc interface ───────────────────────────────────────────────────────────

def _csr_to_petsc(A: sp.csr_matrix) -> PETSc.Mat:
    n   = A.shape[0]
    mat = PETSc.Mat().createAIJWithArrays(
        (n, n),
        (A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.copy()),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat


def run_ksp(
    A:        sp.csr_matrix,
    b:        np.ndarray,
    ksp_type: str,
    pc_type:  str = "none",
) -> tuple[bool, int, float]:
    """
    Solve A x = b with the given PETSc KSP type on a single process.

    Returns (converged, iterations, wall_time).
    wall_time is inf on error; iterations is -1 on error.
    """
    mat = _csr_to_petsc(A)
    x   = mat.createVecRight()
    rhs = mat.createVecLeft()
    rhs.setValues(np.arange(len(b), dtype=np.int32), b.astype(np.float64))
    rhs.assemble()

    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setTolerances(rtol=TOL, atol=1e-50, divtol=1e5, max_it=MAX_ITER)

    t0 = time.perf_counter()
    try:
        ksp.solve(rhs, x)
        elapsed   = time.perf_counter() - t0
        converged = ksp.getConvergedReason() > 0
        iters     = ksp.getIterationNumber()
    except Exception as exc:
        log.debug("KSP %s+%s raised: %s", ksp_type, pc_type, exc)
        elapsed, converged, iters = float("inf"), False, -1
    finally:
        ksp.destroy()
        mat.destroy()
        x.destroy()
        rhs.destroy()

    return converged, iters, elapsed
