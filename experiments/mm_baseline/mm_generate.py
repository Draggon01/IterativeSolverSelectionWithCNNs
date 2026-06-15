"""
mm_generate.py — Generate the MM-AutoSolver baseline dataset.

Produces the same labeled sparse-system samples as the main pipeline
(same matrix types, same PETSc solver benchmarks) but extracts the
17 numerical features and 128×128 density images described in the
MM-AutoSolver paper instead of the main pipeline's 14-feature / 64×64
binary-image format.

Both datasets can then be fed to their respective models for a fair
architectural comparison on identical training matrices.

Output: $DATA_DIR/dataset.h5

Environment variables:
  N_SAMPLES      Number of samples to generate          (default 5000)
  DATA_DIR       Output directory                       (default ./data)
  SEED           NumPy RNG seed                         (default 42)
  MAX_ITER       Max KSP iterations per solver          (default 2000)
  TOL            Relative residual tolerance            (default 1e-8)
  STORE_MATRIX   1 = also store raw CSR matrix data     (default 0)
                 Enables fast image re-rendering without re-running solvers.
"""

import os
import sys
import logging

import h5py
import numpy as np
import scipy.sparse as sp

from generators import (
    sample_matrix, run_ksp,
    random_spd, random_nonsymmetric, poisson_2d, poisson_3d,
)

from mm_model import (
    mm_features, mm_density_image, MM_N_FEATURES, MM_IMAGE_SIZE,
    MM_SOLVER_PAIRS, MM_SOLVER_NAMES, MM_SOLVER_IDX, MM_N_SOLVERS, MM_APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_SAMPLES    = int(os.getenv("N_SAMPLES",   "5000"))
DATA_DIR     = os.getenv("DATA_DIR",       "./data")
SEED         = int(os.getenv("SEED",       "42"))
STORE_MATRIX = os.getenv("STORE_MATRIX", "0") == "1"
FORCE_BUCKET = os.getenv("FORCE_BUCKET")   # 0-7 to lock sampler to one bucket


def shifted_spd(n: int, density: float, rng: "np.random.Generator") -> "sp.csr_matrix":
    """
    Symmetric indefinite matrix: SPD matrix shifted to have negative eigenvalues.
    Classified as 'sym' → enables symmlq+jacobi, symmlq+sor, cr+ilu, bcgsl+*, etc.
    """
    A = random_spd(n, density, rng)
    # Shift by a fraction of the mean diagonal → makes some eigenvalues negative
    shift = float(rng.uniform(0.3, 0.8)) * float(A.diagonal().mean())
    A = A - shift * sp.eye(n, format="csr")
    # Ensure no fully zero rows (PETSc requirement)
    A = A + sp.diags(np.where(np.abs(A.diagonal()) < 1e-10, 1e-4, 0.0))
    return A.tocsr()


def large_nonsymmetric(n: int, density: float, rng: "np.random.Generator") -> "sp.csr_matrix":
    """
    Larger non-symmetric matrix (n up to 5000).
    Enables bcgsl+asm, cgs+gamg, fgmres+gamg on bigger problems.
    """
    return random_nonsymmetric(n, density, rng)


def mm_sample_matrix(rng: "np.random.Generator") -> "tuple[sp.csr_matrix, str]":
    """
    Sampler covering all 19 solver classes.

    Buckets (8 equally weighted):
      0 — small SPD            n ∈ [100, 1000]   → cg+*, cr+*, symmlq+icc
      1 — sym indefinite       n ∈ [100, 2000]   → symmlq+jacobi/sor, cr+ilu, bcgsl+*
      2 — small non-sym        n ∈ [100,  500]   → fbcgsr+*, bcgsl+none, dgmres+none
      3 — large non-sym        n ∈ [1000,5000]   → bcgsl+asm, cgs+gamg, fgmres+gamg
      4 — small Poisson2D     nx ∈ [10,   50]    → cg+*, cr+*
      5 — large Poisson2D     nx ∈ [70,  200]    → GAMG (n up to 40000)
      6 — small Poisson3D     nx ∈ [5,    15]    → cg+*, cr+*
      7 — large Poisson3D     nx ∈ [15,   35]    → GAMG (n up to 42875)
    """
    choice = int(FORCE_BUCKET) if FORCE_BUCKET is not None else int(rng.integers(0, 8))
    if choice == 0:
        n = int(rng.integers(100, 1000))
        d = float(rng.uniform(0.02, 0.08))
        return random_spd(n, d, rng), "spd"
    elif choice == 1:
        n = int(rng.integers(100, 2000))
        d = float(rng.uniform(0.02, 0.08))
        return shifted_spd(n, d, rng), "sym"
    elif choice == 2:
        n = int(rng.integers(100, 500))
        d = float(rng.uniform(0.02, 0.08))
        return random_nonsymmetric(n, d, rng), "nonsym"
    elif choice == 3:
        n = int(rng.integers(5000, 30000))
        # Keep NNZ/row in [10, 40] regardless of n to avoid memory crashes
        nnz_per_row = float(rng.uniform(10, 40))
        d = nnz_per_row / n
        return large_nonsymmetric(n, d, rng), "nonsym"
    elif choice == 4:
        nx = int(rng.integers(10, 50))
        return poisson_2d(nx), "poisson2d"
    elif choice == 5:
        nx = int(rng.integers(70, 200))
        return poisson_2d(nx), "poisson2d"
    elif choice == 6:
        nx = int(rng.integers(5, 15))
        return poisson_3d(nx), "poisson3d"
    else:
        nx = int(rng.integers(15, 35))
        return poisson_3d(nx), "poisson3d"


def mm_best_solver(
    A: "sp.csr_matrix",
    b: "np.ndarray",
    mat_type: str,
) -> "tuple[int | None, np.ndarray]":
    """Run the paper's 19 solver pairs and return (best_label, runtimes)."""
    import scipy.sparse as sp_mod
    all_times: np.ndarray = np.full(MM_N_SOLVERS, np.nan, dtype=np.float32)
    converged: dict[tuple, float] = {}

    for pair in MM_APPLICABLE[mat_type]:
        ksp_type, pc_type = pair
        ok, _iters, t = run_ksp(A, b, ksp_type, pc_type)
        if ok:
            converged[pair] = t
            all_times[MM_SOLVER_IDX[pair]] = float(t)

    if not converged:
        return None, all_times

    best_pair = min(converged, key=converged.__getitem__)
    return int(MM_SOLVER_IDX[best_pair]), all_times


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng      = np.random.default_rng(SEED)
    out_path = os.path.join(DATA_DIR, "dataset.h5")

    from petsc4py import PETSc
    PETSc.Options()["ksp_error_if_not_converged"] = False

    saved = skipped = 0
    label_counts = np.zeros(MM_N_SOLVERS, dtype=int)

    def _open_or_create(path):
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
        if STORE_MATRIX:
            vlen_f32 = h5py.vlen_dtype(np.float32)
            vlen_i32 = h5py.vlen_dtype(np.int32)
            _ensure("mat_data",    shape=(0,),    maxshape=(None,),    dtype=vlen_f32)
            _ensure("mat_indices", shape=(0,),    maxshape=(None,),    dtype=vlen_i32)
            _ensure("mat_indptr",  shape=(0,),    maxshape=(None,),    dtype=vlen_i32)
            _ensure("mat_shape",   shape=(0, 2),  maxshape=(None, 2),  dtype="i4", chunks=(256, 2))
            if "has_matrix_data" not in f.attrs: f.attrs["has_matrix_data"] = True
        return f

    with _open_or_create(out_path) as f:
        n_before = len(f["labels"])
        log.info("Dataset at %s — %d existing samples, adding %d more.",
                 out_path, n_before, N_SAMPLES)

        if STORE_MATRIX:
            log.info("STORE_MATRIX=1 — raw CSR data will be saved alongside each sample.")

        while saved < N_SAMPLES:
            A, mat_type = mm_sample_matrix(rng)
            b           = rng.standard_normal(A.shape[0])

            label, solver_times = mm_best_solver(A, b, mat_type)
            if label is None:
                skipped += 1
                log.warning("No solver converged (skipped=%d); trying next.", skipped)
                continue

            n = n_before + saved
            for ds in ("images", "features", "labels", "runtimes", "source"):
                f[ds].resize(n + 1, axis=0)

            f["images"][n]   = mm_density_image(A)
            f["features"][n] = mm_features(A)
            f["labels"][n]   = label
            f["runtimes"][n] = solver_times
            f["source"][n]   = f"synthetic/{mat_type}"

            if STORE_MATRIX:
                csr = A.tocsr()
                for ds_name in ("mat_data", "mat_indices", "mat_indptr", "mat_shape"):
                    f[ds_name].resize(n + 1, axis=0)
                f["mat_data"][n]    = csr.data.astype(np.float32)
                f["mat_indices"][n] = csr.indices.astype(np.int32)
                f["mat_indptr"][n]  = csr.indptr.astype(np.int32)
                f["mat_shape"][n]   = csr.shape

            f.flush()

            label_counts[label] += 1
            saved += 1
            if saved % 100 == 0:
                log.info("Progress  %d / %d  (skipped=%d)", saved, N_SAMPLES, skipped)

    log.info("Saved %d samples → %s  (skipped=%d)", saved, out_path, skipped)
    log.info("Label distribution: %s",
             {MM_SOLVER_NAMES[i]: int(label_counts[i]) for i in range(MM_N_SOLVERS)
              if label_counts[i] > 0})


if __name__ == "__main__":
    main()
