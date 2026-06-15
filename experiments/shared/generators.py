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


def shifted_spd(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Symmetric indefinite matrix: an SPD matrix shifted to introduce negative eigenvalues.
    Classified as 'sym' — enables symmlq, cr, fcg, minres and their preconditioners.
    """
    A     = random_spd(n, density, rng)
    shift = float(rng.uniform(0.3, 0.8)) * float(A.diagonal().mean())
    A     = A - shift * sp.eye(n, format="csr")
    # Ensure no fully zero rows (PETSc requirement)
    A     = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
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
    """
    Uniformly pick one of 8 buckets and return (A, type_name).

    Buckets:
      0 — small SPD          n ∈ [100,  1 000],  d ∈ [0.02, 0.08]
      1 — sym indefinite     n ∈ [100,  2 000],  d ∈ [0.02, 0.08]   → "sym"
      2 — small non-sym      n ∈ [100,  1 000],  d ∈ [0.02, 0.08]
      3 — large non-sym      n ∈ [1 000, 20 000], NNZ/row ∈ [5, 20]
      4 — small Poisson 2-D  nx ∈ [10,   50]  → n up to   2 500
      5 — large Poisson 2-D  nx ∈ [50,  142]  → n up to ~20 000
      6 — small Poisson 3-D  nx ∈ [5,    15]  → n up to   3 375
      7 — large Poisson 3-D  nx ∈ [15,   27]  → n up to ~19 683
    """
    choice = int(rng.integers(0, 8))
    if choice == 0:
        n = int(rng.integers(100, 1_000))
        d = float(rng.uniform(0.02, 0.08))
        return random_spd(n, d, rng), "spd"
    elif choice == 1:
        n = int(rng.integers(100, 2_000))
        d = float(rng.uniform(0.02, 0.08))
        return shifted_spd(n, d, rng), "sym"
    elif choice == 2:
        n = int(rng.integers(100, 1_000))
        d = float(rng.uniform(0.02, 0.08))
        return random_nonsymmetric(n, d, rng), "nonsym"
    elif choice == 3:
        n = int(rng.integers(1_000, 20_000))
        d = float(rng.uniform(5, 20)) / n   # cap NNZ/row to avoid memory blowup
        return random_nonsymmetric(n, d, rng), "nonsym"
    elif choice == 4:
        nx = int(rng.integers(10, 50))
        return poisson_2d(nx), "poisson2d"
    elif choice == 5:
        nx = int(rng.integers(50, 142))      # n up to 141² ≈ 20 000
        return poisson_2d(nx), "poisson2d"
    elif choice == 6:
        nx = int(rng.integers(5, 15))
        return poisson_3d(nx), "poisson3d"
    else:
        nx = int(rng.integers(15, 28))       # n up to 27³ ≈ 19 683
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
