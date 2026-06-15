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


# ── PDE and structured matrix generators ─────────────────────────────────────

def convection_diffusion_2d(
    nx: int, eps: float, bx: float, by: float,
) -> sp.csr_matrix:
    """
    2-D convection-diffusion: -eps*∇²u + bx*du/dx + by*du/dy = f.
    Upwind FD on an nx×nx grid (Dirichlet BCs).
    Non-symmetric when (bx, by) ≠ (0, 0) — enables GMRES+GAMG over CG.
    """
    n = nx * nx
    h = 1.0 / (nx + 1)

    diag  = np.full(n, 4.0*eps/h**2 + (abs(bx) + abs(by))/h)

    ox_lo = np.full(n - 1, -eps/h**2 - max(bx, 0.0)/h)
    ox_lo[nx - 1::nx] = 0.0                          # no x-wrap across rows
    ox_hi = np.full(n - 1, -eps/h**2 + min(bx, 0.0)/h)
    ox_hi[nx - 1::nx] = 0.0

    oy_lo = np.full(n - nx, -eps/h**2 - max(by, 0.0)/h)
    oy_hi = np.full(n - nx, -eps/h**2 + min(by, 0.0)/h)

    return sp.diags(
        [oy_lo, ox_lo, diag, ox_hi, oy_hi],
        [-nx, -1, 0, 1, nx],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def convection_diffusion_3d(
    nx: int, eps: float, bx: float, by: float, bz: float,
) -> sp.csr_matrix:
    """
    3-D convection-diffusion: -eps*∇²u + b·∇u = f.
    Upwind FD on an nx×nx×nx grid. Non-symmetric when b ≠ 0.
    """
    n   = nx ** 3
    nxy = nx * nx
    h   = 1.0 / (nx + 1)

    diag  = np.full(n, 6.0*eps/h**2 + (abs(bx) + abs(by) + abs(bz))/h)

    ox_lo = np.full(n - 1,   -eps/h**2 - max(bx, 0.0)/h)
    ox_lo[nx - 1::nx] = 0.0
    ox_hi = np.full(n - 1,   -eps/h**2 + min(bx, 0.0)/h)
    ox_hi[nx - 1::nx] = 0.0

    oy_lo = np.full(n - nx,  -eps/h**2 - max(by, 0.0)/h)
    oy_hi = np.full(n - nx,  -eps/h**2 + min(by, 0.0)/h)

    oz_lo = np.full(n - nxy, -eps/h**2 - max(bz, 0.0)/h)
    oz_hi = np.full(n - nxy, -eps/h**2 + min(bz, 0.0)/h)

    return sp.diags(
        [oz_lo, oy_lo, ox_lo, diag, ox_hi, oy_hi, oz_hi],
        [-nxy, -nx, -1, 0, 1, nx, nxy],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def anisotropic_poisson_2d(nx: int, eps: float) -> sp.csr_matrix:
    """
    2-D anisotropic Poisson: -eps*d²u/dx² - d²u/dy² = f.
    SPD for eps > 0; high anisotropy (eps ≪ 1) makes ILU struggle while
    GAMG handles the directional stretching well.
    """
    n = nx * nx
    h = 1.0 / (nx + 1)

    diag = np.full(n, (2.0*eps + 2.0) / h**2)
    ox   = np.full(n - 1,  -eps / h**2)
    ox[nx - 1::nx] = 0.0
    oy   = np.full(n - nx, -1.0 / h**2)

    return sp.diags(
        [oy, ox, diag, ox.copy(), oy.copy()],
        [-nx, -1, 0, 1, nx],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def helmholtz_2d(nx: int, k: float) -> sp.csr_matrix:
    """
    2-D Helmholtz: -∇²u - k²u = f.
    Symmetric; indefinite when k² exceeds the smallest eigenvalue of -∇²
    (roughly 2π² for large nx — so k > ~4.4 typically triggers indefiniteness).
    Enables MINRES, SYMMLQ and other symmetric-indefinite solvers.
    """
    n = nx * nx
    h = 1.0 / (nx + 1)

    diag = np.full(n, 4.0/h**2 - k**2)
    ox   = np.full(n - 1,  -1.0/h**2)
    ox[nx - 1::nx] = 0.0
    oy   = np.full(n - nx, -1.0/h**2)

    return sp.diags(
        [oy, ox, diag, ox.copy(), oy.copy()],
        [-nx, -1, 0, 1, nx],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def random_banded(n: int, bandwidth: int, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Non-symmetric banded matrix with the given bandwidth.
    Upper and lower off-diagonals are drawn independently → non-symmetric.
    Diagonal is chosen for strict diagonal dominance.
    """
    diags_data: list[np.ndarray] = []
    offsets:    list[int]        = []
    row_abs = np.zeros(n)

    for k in range(1, bandwidth + 1):
        size = n - k
        lo = rng.uniform(-1.0, 1.0, size)
        hi = rng.uniform(-1.0, 1.0, size)
        diags_data += [lo, hi]
        offsets    += [-k, k]
        row_abs[k:]    += np.abs(lo)
        row_abs[:size] += np.abs(hi)

    diags_data.append(row_abs + rng.uniform(0.5, 1.5, n))
    offsets.append(0)

    return sp.diags(
        diags_data, offsets, shape=(n, n), format="csr", dtype=np.float64,
    )


# Sampling weights for the 17 buckets in sample_matrix.
# Unnormalised integers — divided below so the sum is always exactly 1.
#
# High weight on large Poisson and large non-sym so that GAMG/ASM-based
# solvers win often enough to share the dataset with CG-family solvers.
# Convection-diffusion and Helmholtz add PDE structure that favours
# GMRES+GAMG and MINRES/SYMMLQ respectively.
_BUCKET_WEIGHTS = np.array([
    3,   #  0 — small SPD                 → cg+ilu, cg+eisenstat, cg+bjacobi
    4,   #  1 — sym indefinite small      → symmlq+*, cr+*
    4,   #  2 — sym indefinite large      → symmlq+*, cr+*, minres+gamg
    3,   #  3 — nonsym small              → fbcgsr+jacobi, bcgsl+none
    5,   #  4 — nonsym medium             → fbcgsr+ilu, dgmres+none
    7,   #  5 — nonsym large              → bcgsl+asm, cgs+gamg, fgmres+gamg
    3,   #  6 — Poisson 2-D small         → cg+ilu, cg+eisenstat
    8,   #  7 — Poisson 2-D large         → minres+gamg, fcg+gamg
    3,   #  8 — Poisson 3-D small         → cg+ilu, cg+eisenstat
    8,   #  9 — Poisson 3-D large         → minres+gamg, fcg+gamg
    5,   # 10 — conv-diff 2-D small       → fbcgsr+jacobi, dgmres+none
    8,   # 11 — conv-diff 2-D large       → gmres+gamg, fgmres+gamg, cgs+gamg
    6,   # 12 — conv-diff 3-D large       → gmres+gamg, fgmres+gamg
    4,   # 13 — aniso Poisson 2-D small   → cg+ilu (ILU degrades with anisotropy)
    8,   # 14 — aniso Poisson 2-D large   → minres+gamg, fcg+gamg (GAMG handles anisotropy)
    6,   # 15 — Helmholtz 2-D             → symmlq+*, minres+gamg (indefinite sym)
    4,   # 16 — random banded             → cr+ilu, cg+ilu (banded ILU is exact)
], dtype=np.float64)
_BUCKET_WEIGHTS /= _BUCKET_WEIGHTS.sum()

_BUCKET_NAMES = [
    "spd-small",        #  0
    "sym-indef-small",  #  1
    "sym-indef-large",  #  2
    "nonsym-small",     #  3
    "nonsym-medium",    #  4
    "nonsym-large",     #  5
    "poisson2d-small",  #  6
    "poisson2d-large",  #  7
    "poisson3d-small",  #  8
    "poisson3d-large",  #  9
    "convdiff2d-small", # 10
    "convdiff2d-large", # 11
    "convdiff3d-large", # 12
    "aniso2d-small",    # 13
    "aniso2d-large",    # 14
    "helmholtz2d",      # 15
    "banded",           # 16
]


def sample_matrix(rng: np.random.Generator) -> tuple[sp.csr_matrix, str, str]:
    """
    Pick one of 17 weighted buckets and return (A, type_name, bucket_name).

    type_name controls which solver pairs are benchmarked (via APPLICABLE):
      "spd"      — all 19 pairs
      "sym"      — symmetric-indefinite; excludes cg and icc
      "nonsym"   — general; excludes symmetric-only KSPs and icc
      "poisson2d"— all 19 pairs (SPD with PDE structure)
      "poisson3d"— all 19 pairs
    """
    bucket = int(rng.choice(len(_BUCKET_WEIGHTS), p=_BUCKET_WEIGHTS))
    bn = _BUCKET_NAMES[bucket]

    if bucket == 0:                          # small SPD
        n = int(rng.integers(100, 1_000))
        d = float(rng.uniform(0.02, 0.08))
        return random_spd(n, d, rng), "spd", bn

    elif bucket == 1:                        # sym indefinite small
        n = int(rng.integers(100, 2_000))
        d = float(rng.uniform(0.02, 0.08))
        return shifted_spd(n, d, rng), "sym", bn

    elif bucket == 2:                        # sym indefinite large
        n = int(rng.integers(2_000, 10_000))
        d = float(rng.uniform(5, 20)) / n
        return shifted_spd(n, d, rng), "sym", bn

    elif bucket == 3:                        # nonsym small
        n = int(rng.integers(100, 1_000))
        d = float(rng.uniform(0.02, 0.08))
        return random_nonsymmetric(n, d, rng), "nonsym", bn

    elif bucket == 4:                        # nonsym medium
        n = int(rng.integers(1_000, 5_000))
        d = float(rng.uniform(5, 20)) / n
        return random_nonsymmetric(n, d, rng), "nonsym", bn

    elif bucket == 5:                        # nonsym large
        n = int(rng.integers(5_000, 20_000))
        d = float(rng.uniform(5, 20)) / n
        return random_nonsymmetric(n, d, rng), "nonsym", bn

    elif bucket == 6:                        # Poisson 2-D small
        nx = int(rng.integers(10, 50))
        return poisson_2d(nx), "poisson2d", bn

    elif bucket == 7:                        # Poisson 2-D large → n ≤ 40 000
        nx = int(rng.integers(70, 201))
        return poisson_2d(nx), "poisson2d", bn

    elif bucket == 8:                        # Poisson 3-D small
        nx = int(rng.integers(5, 15))
        return poisson_3d(nx), "poisson3d", bn

    elif bucket == 9:                        # Poisson 3-D large → n ≤ 42 875
        nx = int(rng.integers(15, 36))
        return poisson_3d(nx), "poisson3d", bn

    elif bucket == 10:                       # conv-diff 2-D small
        nx  = int(rng.integers(10, 50))
        eps = float(rng.uniform(0.001, 0.1))
        bx  = float(rng.uniform(-2.0, 2.0))
        by  = float(rng.uniform(-2.0, 2.0))
        return convection_diffusion_2d(nx, eps, bx, by), "nonsym", bn

    elif bucket == 11:                       # conv-diff 2-D large → n ≤ 40 000
        nx  = int(rng.integers(70, 201))
        eps = float(rng.uniform(0.001, 0.05))
        bx  = float(rng.uniform(-2.0, 2.0))
        by  = float(rng.uniform(-2.0, 2.0))
        return convection_diffusion_2d(nx, eps, bx, by), "nonsym", bn

    elif bucket == 12:                       # conv-diff 3-D large → n ≤ 27 000
        nx  = int(rng.integers(10, 31))
        eps = float(rng.uniform(0.001, 0.05))
        bx  = float(rng.uniform(-2.0, 2.0))
        by  = float(rng.uniform(-2.0, 2.0))
        bz  = float(rng.uniform(-2.0, 2.0))
        return convection_diffusion_3d(nx, eps, bx, by, bz), "nonsym", bn

    elif bucket == 13:                       # aniso Poisson 2-D small
        nx  = int(rng.integers(10, 50))
        eps = float(rng.uniform(0.001, 0.1))
        return anisotropic_poisson_2d(nx, eps), "poisson2d", bn

    elif bucket == 14:                       # aniso Poisson 2-D large → n ≤ 40 000
        nx  = int(rng.integers(70, 201))
        eps = float(rng.uniform(0.001, 0.05))
        return anisotropic_poisson_2d(nx, eps), "poisson2d", bn

    elif bucket == 15:                       # Helmholtz 2-D (symmetric indefinite)
        nx  = int(rng.integers(20, 141))
        h   = 1.0 / (nx + 1)
        # k scales with grid so the Helmholtz parameter k·h ∈ [0.3, 1.5]
        k   = float(rng.uniform(0.3, 1.5)) / h
        return helmholtz_2d(nx, k), "sym", bn

    else:                                    # bucket == 16: random banded
        n   = int(rng.integers(500, 20_000))
        bw  = int(rng.integers(1, min(50, n // 4)))
        return random_banded(n, bw, rng), "nonsym", bn


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
