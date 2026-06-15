"""
generate.py — Generate labeled training data for iterative solver selection.

For each sample:
  1. Draw a sparse matrix A (random SPD, random non-symmetric, 2-D or 3-D Poisson).
  2. Draw a random RHS vector b.
  3. Run all applicable PETSc Krylov solvers and record wall-clock convergence times.
  4. Label the sample with the solver that converges first (or skip if none converge).
  5. Extract the sparsity-pattern image and scalar matrix statistics.

Output: $DATA_DIR/dataset.h5

Environment variables (all optional):
  N_SAMPLES      Number of samples to generate          (default 1000)
  DATA_DIR       Output directory                       (default /workspace/data)
  IMAGE_SIZE     Sparsity image resolution              (default 64)
  IMAGE_MODE     binary | density | log_density | magnitude  (default binary)
  MAX_ITER       Max KSP iterations per solver          (default 2000)
  TOL            Relative residual tolerance            (default 1e-8)
  SEED           NumPy RNG seed                         (default 42)
  STORE_MATRIX   1 = also store raw CSR matrix data     (default 0)
                 Enables fast image re-rendering without re-running solvers.
"""

import os
import logging

import numpy as np
import scipy.sparse as sp
import h5py
from petsc4py import PETSc

from generators import (
    random_spd, random_nonsymmetric, poisson_2d, poisson_3d,
    sample_matrix, run_ksp,
)
from model import (
    matrix_features, sparsity_image,
    SOLVER_PAIRS, SOLVER_NAMES, SOLVER_IDX, N_SOLVERS, N_FEATURES, IMAGE_SIZE, IMAGE_MODE,
    APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
N_SAMPLES    = int(os.getenv("N_SAMPLES",    "1000"))
DATA_DIR     = os.getenv("DATA_DIR",         "/workspace/data")
SEED         = int(os.getenv("SEED",         "42"))
STORE_MATRIX = os.getenv("STORE_MATRIX", "0") == "1"


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

        if STORE_MATRIX:
            vlen_f32 = h5py.vlen_dtype(np.float32)
            vlen_i32 = h5py.vlen_dtype(np.int32)
            ds_mdata   = f.create_dataset("mat_data",   shape=(0,), maxshape=(None,), dtype=vlen_f32)
            ds_mindices= f.create_dataset("mat_indices", shape=(0,), maxshape=(None,), dtype=vlen_i32)
            ds_mindptr = f.create_dataset("mat_indptr",  shape=(0,), maxshape=(None,), dtype=vlen_i32)
            ds_mshape  = f.create_dataset("mat_shape",   shape=(0, 2), maxshape=(None, 2), dtype="i4", chunks=(256, 2))
            f.attrs["has_matrix_data"] = True
            log.info("STORE_MATRIX=1 — raw CSR data will be saved alongside each sample.")

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

            if STORE_MATRIX:
                csr = A.tocsr()
                for ds in (ds_mdata, ds_mindices, ds_mindptr, ds_mshape):
                    ds.resize(saved + 1, axis=0)
                ds_mdata[saved]    = csr.data.astype(np.float32)
                ds_mindices[saved] = csr.indices.astype(np.int32)
                ds_mindptr[saved]  = csr.indptr.astype(np.int32)
                ds_mshape[saved]   = csr.shape

            label_counts[label] += 1
            saved += 1
            if saved % 200 == 0:
                log.info("Progress  %d / %d  (skipped=%d)", saved, N_SAMPLES, skipped)

    log.info("Saved %d samples to %s  (skipped=%d)", saved, out_path, skipped)
    log.info("Label distribution: %s",
             {SOLVER_NAMES[i]: int(label_counts[i]) for i in range(N_SOLVERS)})


if __name__ == "__main__":
    main()
