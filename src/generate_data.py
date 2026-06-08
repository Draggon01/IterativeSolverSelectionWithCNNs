"""
generate_data.py — Generate labeled training data for iterative solver selection.

For each sample:
  1. Draw a sparse matrix A (random SPD, random non-symmetric, 2-D or 3-D Poisson).
  2. Draw a random RHS vector b.
  3. Run all applicable PETSc Krylov solvers and record wall-clock convergence times.
  4. Label the sample with the solver that converges first (or skip if none converge).
  5. Extract the sparsity-pattern image and scalar matrix statistics.

Output: $DATA_DIR/dataset.h5

Environment variables (all optional):
  N_SAMPLES    Number of samples to generate  (default 1 000)
  DATA_DIR     Output directory               (default /workspace/data)
  IMAGE_SIZE   Sparsity image resolution      (default 64)
  MAX_ITER     Max KSP iterations per solver  (default 2 000)
  TOL          Relative residual tolerance    (default 1e-8)
  SEED         NumPy RNG seed                 ($DATA_DIR/dataset.h5
default 42)
"""

import os
import time
import logging

import numpy as np
import scipy.sparse as sp
import h5py
from petsc4py import PETSc

from model import (
    matrix_features, sparsity_image,
    SOLVER_PAIRS, SOLVER_NAMES, SOLVER_IDX, N_SOLVERS, N_FEATURES, IMAGE_SIZE, IMAGE_MODE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
DATA_DIR  = os.getenv("DATA_DIR",      "/workspace/data")
MAX_ITER  = int(os.getenv("MAX_ITER",  "2000"))
TOL       = float(os.getenv("TOL",     "1e-8"))
SEED      = int(os.getenv("SEED",      "42"))

# (KSP, PC) pairs valid for each matrix type.
# SPD: all 30 pairs (CG+ICC and MINRES+ICC are valid).
# Non-symmetric: CG and MINRES excluded; ICC excluded (requires SPD).
_SPD_PAIRS = SOLVER_PAIRS
_GEN_PAIRS = [p for p in SOLVER_PAIRS
              if p[0] in {"gmres", "bicg", "bcgs", "tfqmr"} and p[1] != "icc"]

APPLICABLE: dict[str, list[tuple[str, str]]] = {
    "spd":       _SPD_PAIRS,
    "poisson2d": _SPD_PAIRS,
    "poisson3d": _SPD_PAIRS,
    "nonsym":    _GEN_PAIRS,
}


# ── matrix generators ─────────────────────────────────────────────────────────

def random_spd(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Random symmetric positive-definite matrix (diagonally dominant construction)."""
    A = sp.random(n, n, density=density, format="csr",
                  random_state=rng, dtype=np.float64)
    A = A + A.T
    # Shift diagonal so A is strictly diagonally dominant → guaranteed SPD
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
    oh[nx - 1::nx] = 0.0          # zero out connections that cross row boundaries
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
    A: sp.csr_matrix,
    b: np.ndarray,
    ksp_type: str,
    pc_type: str = "none",
) -> tuple[bool, int, float]:
    """
    Solve A x = b with the given PETSc KSP type on a single process.

    Returns:
        converged  — True if the solver reached the tolerance
        iterations — number of iterations taken (-1 on error)
        wall_time  — seconds elapsed (inf on error)
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
        log.debug("KSP %s raised: %s", ksp_type, exc)
        elapsed, converged, iters = float("inf"), False, -1
    finally:
        ksp.destroy()
        mat.destroy()
        x.destroy()
        rhs.destroy()

    return converged, iters, elapsed


def best_solver_label(
    A: sp.csr_matrix,
    b: np.ndarray,
    mat_type: str,
) -> tuple[int | None, np.ndarray, np.ndarray]:
    """
    Run all applicable solvers and return (label, runtimes, top3) where:
      label    — SOLVER_IDX of the fastest converging solver, or None
      runtimes — float32 (N_SOLVERS,); NaN where solver did not converge
      top3     — int8 (3,); indices of the 3 fastest converging solvers,
                 ranked by wall time; -1 where fewer than k solvers converged
    """
    all_times: np.ndarray = np.full(N_SOLVERS, np.nan, dtype=np.float32)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng      = np.random.default_rng(SEED)
    out_path = os.path.join(DATA_DIR, "dataset.h5")

    # Tell PETSc not to hard-abort on non-convergence
    PETSc.Options()["ksp_error_if_not_converged"] = False

    saved = skipped = 0
    label_counts = np.zeros(N_SOLVERS, dtype=int)

    with h5py.File(out_path, "w") as f:
        ds_img  = f.create_dataset(
            "images",   shape=(0, IMAGE_SIZE, IMAGE_SIZE),
            maxshape=(None, IMAGE_SIZE, IMAGE_SIZE),
            dtype="f4", chunks=(64, IMAGE_SIZE, IMAGE_SIZE),
        )
        ds_feat = f.create_dataset(
            "features", shape=(0, N_FEATURES),
            maxshape=(None, N_FEATURES),
            dtype="f4", chunks=(256, N_FEATURES),
        )
        ds_lbl = f.create_dataset(
            "labels", shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(256,),
        )
        ds_times = f.create_dataset(
            "runtimes", shape=(0, N_SOLVERS), maxshape=(None, N_SOLVERS),
            dtype="f4", chunks=(256, N_SOLVERS),
        )
        ds_src = f.create_dataset(
            "source", shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(), chunks=(256,),
        )
        ds_top3 = f.create_dataset(
            "top3_labels", shape=(0, 3), maxshape=(None, 3),
            dtype="i1", chunks=(256, 3),
        )
        f.attrs["solvers"]    = SOLVER_NAMES
        f.attrs["image_mode"] = IMAGE_MODE

        while saved < N_SAMPLES:
            A, mat_type = sample_matrix(rng)
            b           = rng.standard_normal(A.shape[0])

            label, solver_times, top3 = best_solver_label(A, b, mat_type)
            if label is None:
                skipped += 1
                log.warning("No solver converged (skipped=%d); trying next sample.", skipped)
                continue

            # Grow datasets and append
            for ds in (ds_img, ds_feat, ds_lbl, ds_times, ds_src, ds_top3):
                ds.resize(saved + 1, axis=0)
            ds_img[saved]    = sparsity_image(A)
            ds_feat[saved]   = matrix_features(A)
            ds_lbl[saved]    = label
            ds_times[saved]  = solver_times
            ds_src[saved]    = f"synthetic/{mat_type}"
            ds_top3[saved]   = top3

            label_counts[label] += 1
            saved += 1
            if saved % 200 == 0:
                log.info("Progress  %d / %d  (skipped=%d)", saved, N_SAMPLES, skipped)

    log.info("Saved %d samples to %s  (skipped=%d)", saved, out_path, skipped)
    log.info("Label distribution: %s",
             {SOLVER_NAMES[i]: int(label_counts[i]) for i in range(N_SOLVERS)})


if __name__ == "__main__":
    main()
