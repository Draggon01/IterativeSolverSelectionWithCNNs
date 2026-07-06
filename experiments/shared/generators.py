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
    Symmetric indefinite matrix: an SPD matrix shifted to introduce a small number
    of negative eigenvalues.  Shift is kept to 5–15% of the mean diagonal so the
    matrix remains mildly indefinite and MINRES/SYMMLQ can still converge.
    Classified as 'sym'.
    """
    A     = random_spd(n, density, rng)
    shift = float(rng.uniform(0.05, 0.15)) * float(A.diagonal().mean())
    A     = A - shift * sp.eye(n, format="csr")
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


def shifted_poisson_2d(nx: int, shift_factor: float) -> sp.csr_matrix:
    """
    2-D Poisson shifted below λ_min to be symmetric indefinite.

    The 5-point stencil has the 'consistent ordering' property that makes SOR
    sweep-by-sweep updates highly effective — SOR converges in O(1/h) iterations
    vs Jacobi's O(1/h²).  Shifting past λ_min makes the system indefinite so CG
    fails and SYMMLQ is required → symmlq+sor dominates on these.

    shift_factor controls how far past indefiniteness: 0.1 = barely, 3.0 = moderate.
    """
    A = poisson_2d(nx)
    lam_min = 4.0 - 4.0 * np.cos(np.pi / (nx + 1))  # smallest eigenvalue of discrete -∇²
    shift = (1.0 + shift_factor) * lam_min
    n = nx * nx
    return (A - shift * sp.eye(n, format="csr")).tocsr()


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


def sym_banded_indefinite(n: int, bandwidth: int, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Symmetric banded indefinite matrix: symmetric banded SPD structure shifted
    slightly to introduce a few negative eigenvalues.
    SOR sweep aligns with banded structure → symmlq+sor wins over symmlq+jacobi.
    Classified as 'sym'.
    """
    diags_data: list[np.ndarray] = []
    offsets: list[int] = []
    row_abs = np.zeros(n)
    for k in range(1, bandwidth + 1):
        size = n - k
        vals = rng.uniform(-1.0, 1.0, size)
        diags_data += [vals, vals.copy()]
        offsets    += [-k, k]
        row_abs[k:]    += np.abs(vals)
        row_abs[:size] += np.abs(vals)
    diag = row_abs + rng.uniform(0.5, 1.5, n)
    diags_data.append(diag)
    offsets.append(0)
    A = sp.diags(diags_data, offsets, shape=(n, n), format="csr", dtype=np.float64)
    # Moderate shift: 40–80% of mean diagonal → genuinely indefinite.
    # CR diverges on indefinite systems; SYMMLQ+SOR is robust.
    shift = float(rng.uniform(0.40, 0.80)) * float(A.diagonal().mean())
    A = A - shift * sp.eye(n, format="csr")
    A = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
    return A.tocsr()


def convection_diffusion_2d_moderate(
    nx: int, eps: float, bx: float, by: float
) -> sp.csr_matrix:
    """
    Convection-diffusion 2D with moderate Peclet number (eps=0.1–0.8).
    GAMG works reliably at moderate convection → cgs+gamg and fgmres+gamg compete.
    """
    return convection_diffusion_2d(nx, eps, bx, by)


def barely_dominant_nonsym(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Non-symmetric with barely-above-1 diagonal dominance ratio.
    Off-diagonal entries are large relative to the diagonal, so ILU's incomplete
    factorisation accumulates large errors → ILU-preconditioned solvers stall.
    DGMRES deflates the resulting eigenvalue outliers without a preconditioner.
    """
    A = sp.random(n, n, density=density, format="csr",
                  random_state=rng, dtype=np.float64)
    A.data *= rng.uniform(0.5, 2.0, len(A.data))
    row_sum = np.abs(A).sum(axis=1).A1
    # Diagonal just 1–5% above row sum — barely dominant, poor ILU quality
    diag = row_sum * rng.uniform(1.01, 1.05, n) + 1e-6
    return (A + sp.diags(diag, format="csr")).tocsr()


def sym_indef_deep(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Random (unstructured) symmetric DEEPLY indefinite matrix.

    Uses random_spd as base but shifts by 30-60% of mean diagonal, creating a
    large fraction of negative eigenvalues.  The random sparsity pattern has no
    consistent ordering → SOR's sweep gains nothing over Jacobi.  The deep
    indefiniteness destabilises CR (conjugate residuals) faster than SYMMLQ →
    symmlq+jacobi wins over both cr+jacobi and symmlq+sor.
    """
    A = random_spd(n, density, rng)
    shift = float(rng.uniform(0.30, 0.60)) * float(A.diagonal().mean())
    A = A - shift * sp.eye(n, format="csr")
    A = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
    return A.tocsr()


def anisotropic_poisson_3d(nx: int, eps: float) -> sp.csr_matrix:
    """
    3-D anisotropic Poisson: -eps*d²u/dx² - eps*d²u/dy² - d²u/dz² = f.

    SPD; high anisotropy (eps ≪ 1) makes ILU and SOR fail — GAMG is required.
    For medium-large SPD systems, FCG (CG-type with flexible preconditioner)
    needs fewer operations per iteration than MINRES → fcg+gamg wins.
    Follows the same stencil convention as poisson_3d (no explicit BC zeroing).
    """
    n   = nx ** 3
    nxy = nx * nx
    h   = 1.0 / (nx + 1)

    return sp.diags(
        [
            np.full(n - nxy, -1.0 / h**2),        # z-coupling  (full weight)
            np.full(n - nx,  -1.0 / h**2),         # y-coupling  (full weight)
            np.full(n - 1,   -eps  / h**2),         # x-coupling  (small, anisotropic)
            np.full(n,       (2*eps + 4.0) / h**2), # diagonal
            np.full(n - 1,   -eps  / h**2),
            np.full(n - nx,  -1.0 / h**2),
            np.full(n - nxy, -1.0 / h**2),
        ],
        [-nxy, -nx, -1, 0, 1, nx, nxy],
        shape=(n, n), format="csr", dtype=np.float64,
    )


def sym_banded_mild(n: int, bandwidth: int, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Symmetric banded MILDLY indefinite (shift 10–25% of mean diagonal).

    Bucket 19 (sym_banded_indefinite) uses 40–80% shift, which can make the
    matrix too ill-conditioned for SOR to work reliably as a preconditioner.
    This variant stays close enough to SPD that SOR's sweep still converges
    efficiently, while the mild indefiniteness means CG fails and SYMMLQ is
    required.  The banded ordering gives SOR a clear advantage over Jacobi →
    symmlq+sor wins over symmlq+jacobi and cr+jacobi.
    """
    diags_data: list[np.ndarray] = []
    offsets: list[int] = []
    row_abs = np.zeros(n)
    for k in range(1, bandwidth + 1):
        size = n - k
        vals = rng.uniform(-1.0, 1.0, size)
        diags_data += [vals, vals.copy()]
        offsets    += [-k, k]
        row_abs[k:]    += np.abs(vals)
        row_abs[:size] += np.abs(vals)
    diag = row_abs + rng.uniform(0.5, 1.5, n)
    diags_data.append(diag)
    offsets.append(0)
    A = sp.diags(diags_data, offsets, shape=(n, n), format="csr", dtype=np.float64)
    # Mild shift: only 10–25% of mean diagonal → barely indefinite, SOR still effective
    shift = float(rng.uniform(0.10, 0.25)) * float(A.diagonal().mean())
    A = A - shift * sp.eye(n, format="csr")
    A = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
    return A.tocsr()


def sym_banded_indefinite_wide(n: int, bandwidth: int, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Wide-bandwidth symmetric indefinite matrix (bw 15–60).
    SOR's sweep spans a broader band → symmlq+sor more efficient than
    symmlq+jacobi or minres+gamg on these structures.
    """
    diags_data: list[np.ndarray] = []
    offsets: list[int] = []
    row_abs = np.zeros(n)
    for k in range(1, bandwidth + 1):
        size = n - k
        vals = rng.uniform(-1.0, 1.0, size)
        diags_data += [vals, vals.copy()]
        offsets    += [-k, k]
        row_abs[k:]    += np.abs(vals)
        row_abs[:size] += np.abs(vals)
    diag = row_abs + rng.uniform(0.5, 1.5, n)
    diags_data.append(diag)
    offsets.append(0)
    A = sp.diags(diags_data, offsets, shape=(n, n), format="csr", dtype=np.float64)
    shift = float(rng.uniform(0.30, 0.60)) * float(A.diagonal().mean())
    A = A - shift * sp.eye(n, format="csr")
    A = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
    return A.tocsr()


# ── bucket weights and names ──────────────────────────────────────────────────
#
# Unnormalised integers — divided below so the sum is always exactly 1.
# Original 20 buckets (0-19) unchanged; new buckets (20-27) appended.

_BUCKET_WEIGHTS_RAW = np.array([
    3,   #  0 — small SPD                 → cg+ilu, cg+eisenstat, cg+bjacobi
    3,   #  1 — sym indefinite small      → symmlq+*, cr+*
    3,   #  2 — sym indefinite large      → symmlq+*, cr+*, minres+gamg
    3,   #  3 — nonsym small              → fbcgsr+jacobi, bcgsl+none
    5,   #  4 — nonsym medium             → fbcgsr+ilu, dgmres+none
    7,   #  5 — nonsym large              → bcgsl+asm, cgs+gamg, fgmres+gamg
    3,   #  6 — Poisson 2-D small         → cg+ilu, cg+eisenstat
   10,   #  7 — Poisson 2-D large         → minres+gamg, fcg+gamg, cgs+gamg
    3,   #  8 — Poisson 3-D small         → cg+ilu, cg+eisenstat
   10,   #  9 — Poisson 3-D large         → minres+gamg, fcg+gamg, cgs+gamg
    5,   # 10 — conv-diff 2-D small       → fbcgsr+jacobi, dgmres+none
    5,   # 11 — conv-diff 2-D large       → fbcgsr+ilu, bcgsl+asm
    4,   # 12 — conv-diff 3-D large       → fbcgsr+ilu, bcgsl+asm
    4,   # 13 — aniso Poisson 2-D small   → cg+ilu
    8,   # 14 — aniso Poisson 2-D large   → minres+gamg, fcg+gamg
    6,   # 15 — Helmholtz 2-D             → symmlq+*, minres+gamg
    4,   # 16 — random banded             → cr+ilu, cg+ilu
    6,   # 17 — conv-diff 2D moderate     → cgs+gamg, fgmres+gamg
    6,   # 18 — very large Poisson 2D     → fcg+gamg
    5,   # 19 — sym banded indefinite     → symmlq+sor
    # ── new buckets ──────────────────────────────────────────────────────────
    6,   # 20 — dgmres-barely-dom-small   n=200–2000   → dgmres+none
    6,   # 21 — dgmres-barely-dom-large   n=2000–8000  → dgmres+none
    5,   # 22 — dgmres-barely-dom-medium  n=800–3000, higher density → dgmres+none
    6,   # 23 — sym-banded-wide-medium    bw=15–35     → symmlq+sor
    6,   # 24 — sym-banded-wide-large     bw=20–60     → symmlq+sor
    7,   # 25 — poisson3d-xlarge          nx=36–50     → fcg+gamg
    6,   # 26 — convdiff3d-xlarge         nx=20–35     → cgs+gamg
    6,   # 27 — convdiff2d-large-moderate nx=100–250   → cgs+gamg, fgmres+gamg
    # ── symmlq+sor targeted: grid-structured symmetric indefinite ────────────
    8,   # 28 — shifted-poisson2d-small  nx=20–60    → symmlq+sor (grid SOR advantage)
    8,   # 29 — shifted-poisson2d-large  nx=50–150   → symmlq+sor (grid SOR advantage)
    6,   # 30 — helmholtz2d-indef        k always>4.4 → symmlq+sor/minres+gamg
    # ── symmlq+jacobi targeted: random unstructured deep indefinite ──────────
    8,   # 31 — sym-indef-deep-small   n=200–3000   → symmlq+jacobi
    8,   # 32 — sym-indef-deep-large   n=2000–12000 → symmlq+jacobi
    # ── fcg+gamg targeted: medium-large anisotropic 3D (SPD, GAMG needed) ───
    7,   # 33 — aniso3d-medium         nx=12–22     → fcg+gamg
    7,   # 34 — aniso3d-large          nx=18–30     → fcg+gamg
    # ── symmlq+sor targeted: mild shift banded (SOR stays effective) ─────────
    8,   # 35 — sym-banded-mild-small  bw=3–8,  n=800–5000   → symmlq+sor
    8,   # 36 — sym-banded-mild-large  bw=5–12, n=4000–18000 → symmlq+sor
], dtype=np.float64)

_BUCKET_WEIGHTS = _BUCKET_WEIGHTS_RAW / _BUCKET_WEIGHTS_RAW.sum()

_BUCKET_NAMES = [
    "spd-small",               #  0
    "sym-indef-small",         #  1
    "sym-indef-large",         #  2
    "nonsym-small",            #  3
    "nonsym-medium",           #  4
    "nonsym-large",            #  5
    "poisson2d-small",         #  6
    "poisson2d-large",         #  7
    "poisson3d-small",         #  8
    "poisson3d-large",         #  9
    "convdiff2d-small",        # 10
    "convdiff2d-large",        # 11
    "convdiff3d-large",        # 12
    "aniso2d-small",           # 13
    "aniso2d-large",           # 14
    "helmholtz2d",             # 15
    "banded",                  # 16
    "convdiff2d-moderate",     # 17
    "poisson2d-xlarge",        # 18
    "sym-banded-indef",        # 19
    "dgmres-barely-dom-small", # 20
    "dgmres-barely-dom-large", # 21
    "dgmres-barely-dom-medium",# 22
    "sym-banded-wide-medium",  # 23
    "sym-banded-wide-large",   # 24
    "poisson3d-xlarge",        # 25
    "convdiff3d-xlarge",       # 26
    "convdiff2d-large-moderate", # 27
    "shifted-poisson2d-small",  # 28
    "shifted-poisson2d-large",  # 29
    "helmholtz2d-indef",        # 30
    "sym-indef-deep-small",     # 31
    "sym-indef-deep-large",     # 32
    "aniso3d-medium",           # 33
    "aniso3d-large",            # 34
    "sym-banded-mild-small",    # 35
    "sym-banded-mild-large",    # 36
]

N_BUCKETS = len(_BUCKET_NAMES)


def bucket_info() -> tuple[np.ndarray, list[str]]:
    """Return (base_weights, bucket_names) for the sample_matrix distribution."""
    return _BUCKET_WEIGHTS.copy(), list(_BUCKET_NAMES)


def sample_from_bucket(bucket: int, rng: np.random.Generator) -> tuple[sp.csr_matrix, str, str]:
    """
    Generate a matrix from a specific bucket index and return (A, type_name, bucket_name).
    Same logic as sample_matrix but the bucket is pre-chosen by the caller.
    """
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
        nx  = int(rng.integers(20, 100))
        h   = 1.0 / (nx + 1)
        k   = float(rng.uniform(0.1, 0.5)) / h
        return helmholtz_2d(nx, k), "sym", bn

    elif bucket == 16:                       # random banded
        n   = int(rng.integers(500, 20_000))
        bw  = int(rng.integers(1, min(50, n // 4)))
        return random_banded(n, bw, rng), "nonsym", bn

    elif bucket == 17:                       # conv-diff 2D moderate Pe → cgs+gamg
        nx  = int(rng.integers(50, 151))
        eps = float(rng.uniform(0.1, 0.8))
        bx  = float(rng.uniform(-2.0, 2.0))
        by  = float(rng.uniform(-2.0, 2.0))
        return convection_diffusion_2d(nx, eps, bx, by), "nonsym", bn

    elif bucket == 18:                       # very large Poisson 2D → fcg+gamg
        nx  = int(rng.integers(200, 301))   # n ∈ [40000, 90000]
        return poisson_2d(nx), "poisson2d", bn

    elif bucket == 19:                       # sym banded indefinite → symmlq+sor
        n   = int(rng.integers(1_000, 10_000))
        bw  = int(rng.integers(2, 15))
        return sym_banded_indefinite(n, bw, rng), "sym", bn

    # ── new buckets (20-27) ───────────────────────────────────────────────────

    elif bucket == 20:                       # dgmres target: barely dominant, small
        n   = int(rng.integers(200, 2_000))
        d   = float(rng.uniform(0.02, 0.08))
        return barely_dominant_nonsym(n, d, rng), "nonsym", bn

    elif bucket == 21:                       # dgmres target: barely dominant, large
        n   = int(rng.integers(2_000, 8_000))
        d   = float(rng.uniform(5, 15)) / n
        return barely_dominant_nonsym(n, d, rng), "nonsym", bn

    elif bucket == 22:                       # dgmres target: medium, higher density
        n   = int(rng.integers(800, 3_000))
        d   = float(rng.uniform(0.04, 0.12))
        return barely_dominant_nonsym(n, d, rng), "nonsym", bn

    elif bucket == 23:                       # symmlq+sor target: wide band, medium
        n   = int(rng.integers(1_000, 6_000))
        bw  = int(rng.integers(15, 36))
        return sym_banded_indefinite_wide(n, bw, rng), "sym", bn

    elif bucket == 24:                       # symmlq+sor target: wide band, large
        n   = int(rng.integers(3_000, 12_000))
        bw  = int(rng.integers(20, 61))
        return sym_banded_indefinite_wide(n, bw, rng), "sym", bn

    elif bucket == 25:                       # fcg+gamg target: very large 3D Poisson
        nx  = int(rng.integers(36, 51))     # n ∈ [46656, 132651]
        return poisson_3d(nx), "poisson3d", bn

    elif bucket == 26:                       # cgs+gamg target: large 3D conv-diff
        nx  = int(rng.integers(20, 36))
        eps = float(rng.uniform(0.05, 0.4))
        bx  = float(rng.uniform(-1.5, 1.5))
        by  = float(rng.uniform(-1.5, 1.5))
        bz  = float(rng.uniform(-1.5, 1.5))
        return convection_diffusion_3d(nx, eps, bx, by, bz), "nonsym", bn

    elif bucket == 27:                       # large 2D conv-diff moderate
        nx  = int(rng.integers(100, 251))   # n ∈ [10000, 63001]
        eps = float(rng.uniform(0.05, 0.5))
        bx  = float(rng.uniform(-2.0, 2.0))
        by  = float(rng.uniform(-2.0, 2.0))
        return convection_diffusion_2d(nx, eps, bx, by), "nonsym", bn

    elif bucket == 28:                       # shifted Poisson 2D small → symmlq+sor
        nx  = int(rng.integers(20, 60))     # n ∈ [400, 3600]
        sf  = float(rng.uniform(0.1, 3.0))
        return shifted_poisson_2d(nx, sf), "sym", bn

    elif bucket == 29:                       # shifted Poisson 2D large → symmlq+sor
        nx  = int(rng.integers(50, 150))    # n ∈ [2500, 22500]
        sf  = float(rng.uniform(0.1, 2.0))
        return shifted_poisson_2d(nx, sf), "sym", bn

    elif bucket == 30:                       # Helmholtz guaranteed indefinite
        nx  = int(rng.integers(20, 80))
        k   = float(rng.uniform(5.0, 15.0)) # always > 4.44 → always indefinite
        return helmholtz_2d(nx, k), "sym", bn

    elif bucket == 31:                       # random unstructured deep-indef → symmlq+jacobi
        n   = int(rng.integers(200, 3_000))
        d   = float(rng.uniform(0.02, 0.08))
        return sym_indef_deep(n, d, rng), "sym", bn

    elif bucket == 32:                       # same, larger → symmlq+jacobi
        n   = int(rng.integers(2_000, 12_000))
        d   = float(rng.uniform(5, 15)) / n
        return sym_indef_deep(n, d, rng), "sym", bn

    elif bucket == 33:                       # anisotropic 3D Poisson medium → fcg+gamg
        nx  = int(rng.integers(12, 23))     # n ∈ [1728, 12167]
        eps = float(rng.uniform(0.001, 0.05))
        return anisotropic_poisson_3d(nx, eps), "poisson3d", bn

    elif bucket == 34:                       # anisotropic 3D Poisson large → fcg+gamg
        nx  = int(rng.integers(18, 31))     # n ∈ [5832, 27000]
        eps = float(rng.uniform(0.001, 0.03))
        return anisotropic_poisson_3d(nx, eps), "poisson3d", bn

    elif bucket == 35:                       # mild-shift banded indef small → symmlq+sor
        n   = int(rng.integers(800, 5_000))
        bw  = int(rng.integers(3, 9))
        return sym_banded_mild(n, bw, rng), "sym", bn

    else:                                    # bucket == 36: mild-shift banded indef large
        n   = int(rng.integers(4_000, 18_000))
        bw  = int(rng.integers(5, 13))
        return sym_banded_mild(n, bw, rng), "sym", bn


def sample_matrix(rng: np.random.Generator) -> tuple[sp.csr_matrix, str, str]:
    """
    Pick one of the weighted buckets and return (A, type_name, bucket_name).

    type_name controls which solver pairs are benchmarked (via APPLICABLE):
      "spd"      — all 19 pairs
      "sym"      — symmetric-indefinite; excludes cg and icc
      "nonsym"   — general; excludes symmetric-only KSPs and icc
      "poisson2d"— all 19 pairs (SPD with PDE structure)
      "poisson3d"— all 19 pairs
    """
    bucket = int(rng.choice(N_BUCKETS, p=_BUCKET_WEIGHTS))
    return sample_from_bucket(bucket, rng)


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
