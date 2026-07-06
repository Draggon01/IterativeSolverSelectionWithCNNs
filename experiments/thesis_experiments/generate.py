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
import json
import logging
import datetime

import numpy as np
import scipy.sparse as sp
import h5py
from petsc4py import PETSc

from generators import (
    random_spd, random_nonsymmetric, poisson_2d, poisson_3d,
    sample_matrix, sample_from_bucket, bucket_info, N_BUCKETS, run_ksp, MAX_ITER,
)
from model import (
    matrix_features, sparsity_image,
    SOLVER_PAIRS, SOLVER_NAMES, SOLVER_IDX, N_SOLVERS, N_FEATURES, IMAGE_SIZE, IMAGE_MODE,
    APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
N_SAMPLES     = int(os.getenv("N_SAMPLES",    "1000"))
MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "0"))   # 0 = disabled
STATS_FILE    = os.getenv("STATS_FILE", "")            # "" = DATA_DIR/sampler_stats.json
DATA_DIR     = os.getenv("DATA_DIR",         "/workspace/data")
SEED         = int(os.getenv("SEED",         "42"))
STORE_MATRIX = os.getenv("STORE_MATRIX", "0") == "1"
VERBOSE      = os.getenv("VERBOSE", "0") == "1"

if VERBOSE:
    logging.getLogger().setLevel(logging.DEBUG)


# ── adaptive sampler stats ────────────────────────────────────────────────────

class SamplerStats:
    """
    Tracks per-bucket solver win counts across runs and computes biased bucket
    weights to oversample buckets that historically produce underrepresented classes.

    Persisted as JSON so statistics improve across multiple datagen runs.
    Only active when MIN_PER_CLASS > 0; otherwise sampling is unaffected.
    """

    def __init__(self, path: str):
        self.path = path
        # bucket_name -> {solver_idx -> win_count}
        self.bucket_wins: dict[str, dict[int, int]] = {}
        self.total_generated: int = 0

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.bucket_wins = {
                bn: {int(k): v for k, v in wins.items()}
                for bn, wins in data.get("bucket_wins", {}).items()
            }
            self.total_generated = data.get("total_generated", 0)
            log.info("Loaded sampler stats from %s  (%d total generated)",
                     self.path, self.total_generated)
            return True
        except Exception as exc:
            log.warning("Could not load sampler stats from %s: %s — starting fresh.", self.path, exc)
            return False

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        data = {
            "bucket_wins": {
                bn: {str(k): v for k, v in wins.items()}
                for bn, wins in self.bucket_wins.items()
            },
            "total_generated": self.total_generated,
            "last_updated": datetime.datetime.now().isoformat(),
        }
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def update(self, bucket_name: str, solver_idx: int) -> None:
        wins = self.bucket_wins.setdefault(bucket_name, {})
        wins[solver_idx] = wins.get(solver_idx, 0) + 1
        self.total_generated += 1

    def effective_weights(
        self,
        label_counts: np.ndarray,
        min_per_class: int,
        base_weights: np.ndarray,
        bucket_names: list[str],
        alpha: float = 4.0,
        min_history: int = 20,
    ) -> np.ndarray:
        """
        Return adjusted bucket weights that oversample buckets producing short classes.

        Buckets with less than `min_history` observed samples use base weights
        (no bias yet — not enough data to estimate P(class|bucket)).
        `alpha` controls bias strength: effective_weight = base * (1 + alpha * score).
        """
        short = [c for c in range(len(label_counts)) if label_counts[c] < min_per_class]
        if not short or not self.bucket_wins:
            return base_weights.copy()

        scores = np.zeros(len(bucket_names))
        for i, bn in enumerate(bucket_names):
            wins = self.bucket_wins.get(bn, {})
            total_in_bucket = sum(wins.values())
            if total_in_bucket < min_history:
                continue
            for c in short:
                deficit = (min_per_class - label_counts[c]) / max(min_per_class, 1)
                p_c_given_b = wins.get(c, 0) / total_in_bucket
                scores[i] += deficit * p_c_given_b

        if scores.max() > 0:
            scores /= scores.max()

        effective = base_weights * (1.0 + alpha * scores)
        effective /= effective.sum()
        return effective


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
    n = A.shape[0]
    all_times: np.ndarray = np.full(N_SOLVERS, np.nan, dtype=np.float32)
    converged: dict[tuple, float] = {}
    for pair in APPLICABLE[mat_type]:
        ksp_type, pc_type = pair
        ok, iters, t = run_ksp(A, b, ksp_type, pc_type)
        if ok:
            converged[pair] = t
            all_times[SOLVER_IDX[pair]] = float(t)
        if VERBOSE:
            status = "OK    " if ok else ("MAXITER" if iters >= MAX_ITER else "DIVERG ")
            log.info("    %-10s+%-10s  n=%-6d  %s  iters=%-5d  t=%.3fs",
                     ksp_type, pc_type, n, status, iters, t)

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

    log.info("─" * 60)
    log.info("  generate.py — thesis_experiments")
    log.info("  DATA_DIR     : %s", DATA_DIR)
    log.info("  N_SAMPLES    : %d", N_SAMPLES)
    log.info("  IMAGE_MODE   : %s  IMAGE_SIZE: %d", IMAGE_MODE, IMAGE_SIZE)
    log.info("  STORE_MATRIX : %s", STORE_MATRIX)
    log.info("  SEED         : %d", SEED)
    log.info("  MAX_ITER     : %d  TOL: %s", MAX_ITER, os.getenv("TOL", "1e-8"))
    log.info("  VERBOSE      : %s", VERBOSE)
    log.info("  Solvers      : %d  (%s … %s)", N_SOLVERS, SOLVER_NAMES[0], SOLVER_NAMES[-1])
    if MIN_PER_CLASS > 0:
        stats_path = STATS_FILE or os.path.join(DATA_DIR, "sampler_stats.json")
        log.info("  MIN_PER_CLASS: %d  (N_SAMPLES=%d is a hard cap)", MIN_PER_CLASS, N_SAMPLES)
        log.info("  STATS_FILE   : %s", stats_path)
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as _f:
            n_existing = len(_f["labels"])
        log.info("  Existing file: %s  (%d samples) — will APPEND", out_path, n_existing)
    else:
        log.info("  Output       : %s  (new file)", out_path)
    log.info("─" * 60)

    # Tell PETSc not to hard-abort on non-convergence
    PETSc.Options()["ksp_error_if_not_converged"] = False

    saved = skipped = 0
    label_counts = np.zeros(N_SOLVERS, dtype=int)

    # Load persistent sampler stats (only used when MIN_PER_CLASS > 0)
    base_weights, bucket_names = bucket_info()
    stats_path = STATS_FILE or os.path.join(DATA_DIR, "sampler_stats.json")
    sampler_stats = SamplerStats(stats_path)
    if MIN_PER_CLASS > 0:
        sampler_stats.load()

    def _all_classes_met():
        return MIN_PER_CLASS > 0 and np.all(label_counts >= MIN_PER_CLASS)

    def _pick_bucket() -> int:
        if MIN_PER_CLASS == 0:
            return int(rng.choice(N_BUCKETS, p=base_weights))
        weights = sampler_stats.effective_weights(
            label_counts, MIN_PER_CLASS, base_weights, bucket_names,
        )
        return int(rng.choice(N_BUCKETS, p=weights))

    def _open_or_create(path):
        f = h5py.File(path, "a")
        def _ensure(name, **kwargs):
            if name not in f:
                f.create_dataset(name, **kwargs)
        if not STORE_MATRIX:
            _ensure("images",  shape=(0, IMAGE_SIZE, IMAGE_SIZE),
                    maxshape=(None, IMAGE_SIZE, IMAGE_SIZE),
                    dtype="f4", chunks=(64, IMAGE_SIZE, IMAGE_SIZE))
        _ensure("features",    shape=(0, N_FEATURES), maxshape=(None, N_FEATURES),
                dtype="f4", chunks=(256, N_FEATURES))
        _ensure("labels",      shape=(0,), maxshape=(None,), dtype="i4", chunks=(256,))
        _ensure("runtimes",    shape=(0, N_SOLVERS), maxshape=(None, None),
                dtype="f4", chunks=(256, N_SOLVERS))
        _ensure("source",      shape=(0,), maxshape=(None,),
                dtype=h5py.string_dtype(), chunks=(256,))
        _ensure("top3_labels", shape=(0, 3), maxshape=(None, 3),
                dtype="i1", chunks=(256, 3))
        if "solvers"    not in f.attrs: f.attrs["solvers"]    = SOLVER_NAMES
        if "image_mode" not in f.attrs: f.attrs["image_mode"] = IMAGE_MODE
        if STORE_MATRIX:
            vlen_f32 = h5py.vlen_dtype(np.float32)
            vlen_i32 = h5py.vlen_dtype(np.int32)
            _ensure("mat_data",    shape=(0,),   maxshape=(None,),   dtype=vlen_f32)
            _ensure("mat_indices", shape=(0,),   maxshape=(None,),   dtype=vlen_i32)
            _ensure("mat_indptr",  shape=(0,),   maxshape=(None,),   dtype=vlen_i32)
            _ensure("mat_shape",   shape=(0, 2), maxshape=(None, 2), dtype="i4", chunks=(256, 2))
            if "has_matrix_data" not in f.attrs: f.attrs["has_matrix_data"] = True
        return f

    with _open_or_create(out_path) as f:
        n_before = len(f["labels"])

        # Seed label_counts from existing data so MIN_PER_CLASS accounts for prior samples
        if n_before > 0:
            existing_labels = f["labels"][:]
            for lbl in existing_labels:
                label_counts[lbl] += 1
            if MIN_PER_CLASS > 0:
                short = [(SOLVER_NAMES[i], int(label_counts[i]))
                         for i in range(N_SOLVERS) if label_counts[i] < MIN_PER_CLASS]
                log.info("Existing counts loaded. %d/%d classes already meet MIN_PER_CLASS=%d.",
                         N_SOLVERS - len(short), N_SOLVERS, MIN_PER_CLASS)
                if short:
                    log.info("Still short: %s", short)

        log.info("Dataset at %s — %d existing samples, target: %s.",
                 out_path, n_before,
                 f"all classes >= {MIN_PER_CLASS} (cap={N_SAMPLES})" if MIN_PER_CLASS > 0
                 else f"add {N_SAMPLES} more")

        if STORE_MATRIX:
            log.info("STORE_MATRIX=1 — images skipped; run `render` to produce them.")
        else:
            log.info("Images will be rendered inline (mode=%s size=%d).", IMAGE_MODE, IMAGE_SIZE)

        while not _all_classes_met() and saved < N_SAMPLES:
            A, mat_type, bucket = sample_from_bucket(_pick_bucket(), rng)
            b                   = rng.standard_normal(A.shape[0])
            log.info("Trying  #%-5d  bucket=%-18s  type=%-10s  n=%-6d  nnz=%d",
                     n_before + saved + 1, bucket, mat_type, A.shape[0], A.nnz)

            label, solver_times, top3 = best_solver_label(A, b, mat_type)
            if label is None:
                skipped += 1
                log.warning("  → no solver converged  (saved=%d skipped=%d)", saved, skipped)
                continue
            log.info("  → best: %s  (saved=%d skipped=%d)",
                     SOLVER_NAMES[label], saved + 1, skipped)

            n = n_before + saved
            core_datasets = ("features", "labels", "runtimes", "source", "top3_labels")
            for ds in core_datasets:
                f[ds].resize(n + 1, axis=0)
            if not STORE_MATRIX:
                f["images"].resize(n + 1, axis=0)
                f["images"][n] = sparsity_image(A)
            f["features"][n]    = matrix_features(A)
            f["labels"][n]      = label
            f["runtimes"][n]    = solver_times
            f["source"][n]      = f"synthetic/{mat_type}"
            f["top3_labels"][n] = top3

            if STORE_MATRIX:
                csr = A.tocsr()
                for ds in ("mat_data", "mat_indices", "mat_indptr", "mat_shape"):
                    f[ds].resize(n + 1, axis=0)
                f["mat_data"][n]    = csr.data.astype(np.float32)
                f["mat_indices"][n] = csr.indices.astype(np.int32)
                f["mat_indptr"][n]  = csr.indptr.astype(np.int32)
                f["mat_shape"][n]   = csr.shape

            f.flush()

            label_counts[label] += 1
            sampler_stats.update(bucket, label)
            saved += 1
            if MIN_PER_CLASS > 0 and saved % 1000 == 0:
                sampler_stats.save()
                log.info("Sampler stats checkpoint saved → %s", stats_path)
            if saved % 200 == 0:
                if MIN_PER_CLASS > 0:
                    short = [(SOLVER_NAMES[i], int(label_counts[i]))
                             for i in range(N_SOLVERS) if label_counts[i] < MIN_PER_CLASS]
                    met = N_SOLVERS - len(short)
                    log.info("Progress  saved=%d  cap=%d  classes_met=%d/%d  short=%s",
                             saved, N_SAMPLES, met, N_SOLVERS, short)
                else:
                    log.info("Progress  %d / %d  (skipped=%d)", saved, N_SAMPLES, skipped)

    if MIN_PER_CLASS > 0:
        short = [(SOLVER_NAMES[i], int(label_counts[i]))
                 for i in range(N_SOLVERS) if label_counts[i] < MIN_PER_CLASS]
        if short:
            log.warning("Hit N_SAMPLES cap (%d). Classes still below %d: %s",
                        N_SAMPLES, MIN_PER_CLASS, short)
        else:
            log.info("All %d classes have >= %d samples.", N_SOLVERS, MIN_PER_CLASS)
        sampler_stats.save()
        log.info("Sampler stats saved → %s", stats_path)

    log.info("Saved %d samples → %s  (skipped=%d, total=%d)",
             saved, out_path, skipped, n_before + saved)
    log.info("Label distribution (total): %s",
             {SOLVER_NAMES[i]: int(label_counts[i]) for i in range(N_SOLVERS)
              if label_counts[i] > 0})


if __name__ == "__main__":
    main()
