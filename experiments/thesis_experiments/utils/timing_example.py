#!/usr/bin/env python3
"""
timing_example.py — Benchmark two matrices with two model configurations:

  Matrix 1: SPD (Poisson 2D ~100k)    → Campaign 3 model: rcm_signed_magnitude__density_64
  Matrix 2: Nonsymmetric random ~100k  → Campaign 4 model: magnitude__rcm_log_density_128

For each matrix:
  - Run all 19 PETSc solvers (converging + DNF)
  - Time feature extraction (14 and 20 features)
  - Time image rendering (all 14 modes, 64px and 128px)
  - Compare model pipeline vs random solver selection

Run via Docker:
  docker compose run --rm timing
"""

import time

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from model import matrix_features, sparsity_image, SOLVER_PAIRS, SOLVER_NAMES
from generators import run_ksp, random_nonsymmetric, poisson_2d

PETSc.Options()["ksp_error_if_not_converged"] = False

# ── Matrix parameters ─────────────────────────────────────────────────────────
NNZ_PER_ROW = 50     # off-diagonal non-zeros per row for nonsymmetric matrices

GRID_SIZE_50K  = 224  # 224×224 = 50,176
GRID_SIZE_100K = 317  # 317×317 = 100,489

# ── Model configurations ──────────────────────────────────────────────────────
MODEL_C3 = dict(
    label    = "C3: rcm_signed_magnitude__density_64",
    modes    = ["rcm_signed_magnitude", "density"],
    size     = 64,
    feat_n   = 14,
    infer_ms = 2.8,
    mrt      = 1.063,
)
MODEL_C4 = dict(
    label    = "C4: magnitude__rcm_log_density_128",
    modes    = ["magnitude", "rcm_log_density"],
    size     = 128,
    feat_n   = 20,
    infer_ms = 10.9,
    mrt      = 1.051,
)

IMAGE_MODES = [
    "binary", "density", "log_density", "magnitude", "sign",
    "signed_magnitude", "symmetry", "diagonal",
    "rcm_binary", "rcm_density", "rcm_log_density", "rcm_magnitude",
    "rcm_sign", "rcm_signed_magnitude",
]


# ── Feature extraction ────────────────────────────────────────────────────────

def matrix_features_14(A: sp.csr_matrix) -> np.ndarray:
    """Compute only the original 14 features (campaigns 1 and 3)."""
    n, nnz    = A.shape[0], A.nnz
    frob      = sp.linalg.norm(A, "fro")
    sym_norm  = sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12)
    diag_vals = A.diagonal()
    diag_abs  = np.abs(diag_vals)
    offsum    = np.array(np.abs(A).sum(axis=1)).ravel() - diag_abs
    dom       = float(np.mean(diag_abs / (offsum + 1e-12)))
    vals      = A.data if nnz > 0 else np.array([0.0])
    rng = np.random.default_rng(0)
    v   = rng.standard_normal(n)
    v  /= np.linalg.norm(v) + 1e-12
    for _ in range(8):
        w  = A @ v
        nw = float(np.linalg.norm(w))
        if nw < 1e-12:
            break
        v = w / nw
    spectral_rad = float(abs(v @ (A @ v))) / (float(v @ v) + 1e-12)
    diag_nz      = diag_abs[diag_abs > 1e-14]
    cond_est     = float(diag_nz.max() / diag_nz.min()) if len(diag_nz) >= 2 else 1.0
    coo          = A.tocoo()
    bandwidth    = (float(np.abs(coo.row.astype(np.int64) - coo.col.astype(np.int64)).max()) / n
                    if nnz > 0 else 0.0)
    diag_nnz_frac = float(np.sum(diag_abs > 1e-12)) / n
    row_norms     = np.sqrt(np.array(A.power(2).sum(axis=1)).ravel())
    row_norm_cv   = float(row_norms.std() / (row_norms.mean() + 1e-12))
    offdiag_frob  = float(sp.linalg.norm(A - sp.diags(diag_vals), "fro")) / (frob + 1e-12)
    return np.array([
        np.log1p(n), np.log1p(nnz), nnz / (n * n), float(sym_norm),
        float(np.clip(dom, 0.0, 20.0)), frob / (n + 1e-12),
        float(diag_vals.sum()) / n,
        float(np.abs(vals).max()) / (float(np.abs(vals).mean()) + 1e-12),
        np.log1p(spectral_rad), np.log1p(cond_est), bandwidth,
        diag_nnz_frac, float(np.clip(row_norm_cv, 0.0, 20.0)), offdiag_frob,
    ], dtype=np.float32)


def time_features_individually(A: sp.csr_matrix, n_reps: int = 5) -> list[tuple[str, str, float]]:
    """Time each feature's marginal cost with shared intermediates pre-computed."""
    n, nnz    = A.shape[0], A.nnz
    frob      = sp.linalg.norm(A, "fro")
    diag_vals = A.diagonal()
    diag_abs  = np.abs(diag_vals)
    offsum    = np.array(np.abs(A).sum(axis=1)).ravel() - diag_abs
    dom_ratios= diag_abs / (offsum + 1e-12)
    vals      = A.data if nnz > 0 else np.array([0.0])
    coo       = A.tocoo()
    row_norms = np.sqrt(np.array(A.power(2).sum(axis=1)).ravel())
    diag_nz   = diag_abs[diag_abs > 1e-14]

    def T(fn):
        t0 = time.perf_counter()
        for _ in range(n_reps):
            fn()
        return (time.perf_counter() - t0) / n_reps * 1000

    return [
        ("0",  "log(1+n)",                  T(lambda: np.log1p(n))),
        ("1",  "log(1+nnz)",                T(lambda: np.log1p(nnz))),
        ("2",  "density (nnz/n²)",          T(lambda: nnz / (n * n))),
        ("3",  "asymmetry ‖A−Aᵀ‖/‖A‖",     T(lambda: sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12))),
        ("4",  "diag dominance mean",        T(lambda: float(np.mean(dom_ratios)))),
        ("5",  "Frobenius norm / n",         T(lambda: sp.linalg.norm(A, "fro") / (n + 1e-12))),
        ("6",  "trace / n",                  T(lambda: float(diag_vals.sum()) / n)),
        ("7",  "max / mean |entry|",         T(lambda: float(np.abs(vals).max()) / (float(np.abs(vals).mean()) + 1e-12))),
        ("8",  "spectral radius (8 iter)",   T(lambda: _spectral_radius(A, n))),
        ("9",  "diag cond proxy",            T(lambda: float(diag_nz.max() / diag_nz.min()) if len(diag_nz) >= 2 else 1.0)),
        ("10", "bandwidth / n",              T(lambda: float(np.abs(coo.row.astype(np.int64) - coo.col.astype(np.int64)).max()) / n)),
        ("11", "diag nnz fraction",          T(lambda: float(np.sum(diag_abs > 1e-12)) / n)),
        ("12", "row norm CV",                T(lambda: float(row_norms.std() / (row_norms.mean() + 1e-12)))),
        ("13", "off-diag Frobenius frac",    T(lambda: float(sp.linalg.norm(A - sp.diags(diag_vals), "fro")) / (frob + 1e-12))),
        ("14", "neg off-diag fraction",      T(lambda: _neg_offdiag(coo))),
        ("15", "pos diag fraction",          T(lambda: float(np.sum(diag_vals > 1e-14)) / n)),
        ("16", "structural symmetry",        T(lambda: _struct_sym(coo))),
        ("17", "min diag dominance",         T(lambda: float(np.clip(dom_ratios.min(), 0.0, 20.0)))),
        ("18", "nnz-per-row CV",             T(lambda: float(np.clip(np.diff(A.indptr).std() / (np.diff(A.indptr).mean() + 1e-12), 0, 20)))),
        ("19", "diag dom variance",          T(lambda: float(np.clip(dom_ratios.var(), 0.0, 20.0)))),
    ]


def _spectral_radius(A, n):
    rng = np.random.default_rng(0)
    v   = rng.standard_normal(n)
    v  /= np.linalg.norm(v) + 1e-12
    for _ in range(8):
        w  = A @ v
        nw = float(np.linalg.norm(w))
        if nw < 1e-12:
            break
        v = w / nw
    return float(abs(v @ (A @ v))) / (float(v @ v) + 1e-12)


def _neg_offdiag(coo):
    d = coo.data[coo.row != coo.col]
    return float(np.sum(d < 0)) / (len(d) + 1e-12)


def _struct_sym(coo):
    pairs     = set(zip(coo.row.tolist(), coo.col.tolist()))
    n_sym     = sum(1 for (r, c) in pairs if r != c and (c, r) in pairs)
    n_offdiag = sum(1 for (r, c) in pairs if r != c)
    return float(n_sym) / (n_offdiag + 1e-12)


# ── Solver benchmarking ───────────────────────────────────────────────────────

def benchmark_solvers(A: sp.csr_matrix) -> list[tuple[str, float, bool]]:
    """Returns (solver_name, elapsed_ms, converged) for all 19 solvers."""
    b = np.ones(A.shape[0])
    results = []
    for (ksp, pc), name in zip(SOLVER_PAIRS, SOLVER_NAMES):
        converged, _, elapsed = run_ksp(A, b, ksp, pc)
        results.append((name, elapsed * 1000, converged))
    return results


# ── Per-matrix benchmark ──────────────────────────────────────────────────────

def run_matrix_benchmark(mat_label: str, A: sp.csr_matrix, model: dict,
                         solver_results: list | None = None,
                         feat_rows: list | None = None,
                         render_64: dict | None = None,
                         render_128: dict | None = None):
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  {mat_label}  |  {model['label']}")
    print(f"{sep}\n")

    n = A.shape[0]
    print(f"Matrix: n={n:,},  nnz={A.nnz:,}\n")

    # Solver runtimes (reuse if pre-computed)
    if solver_results is None:
        print("Running all 19 solvers ...", flush=True)
        solver_results = benchmark_solvers(A)
    n_conv = sum(1 for _, _, c in solver_results if c)
    n_dnf  = 19 - n_conv
    print(f"{n_conv}/19 converged,  {n_dnf} DNF\n")

    converging = [(name, ms) for name, ms, c in solver_results if c]
    if not converging:
        print("No solvers converged — skipping.\n")
        return None
    best_ms   = min(ms for _, ms in converging)
    best_name = next(name for name, ms in converging if ms == best_ms)

    print(f"  {'Solver':<30} {'Time (ms)':>10}")
    print(f"  {'-'*42}")
    for name, ms, conv in sorted(solver_results, key=lambda x: (not x[2], x[1])):
        if not conv:
            tag = "  DNF"
        elif ms >= 10_000:
            tag = "  SLOW (≥10s, treated as failure)"
        else:
            tag = ""
        print(f"  {name:<30} {ms:>10.2f}{tag}")
    print(f"\n  Best: {best_name} ({best_ms:.2f} ms)\n")

    # Feature extraction (reuse if pre-computed)
    if feat_rows is None:
        N_REPS = 10
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            matrix_features_14(A)
        feat14_ms = (time.perf_counter() - t0) / N_REPS * 1000

        t0 = time.perf_counter()
        for _ in range(N_REPS):
            matrix_features(A)
        feat20_ms = (time.perf_counter() - t0) / N_REPS * 1000

        print(f"Feature extraction (avg over {N_REPS} runs):")
        print(f"  14 features (C1/C3):  {feat14_ms:.2f} ms")
        print(f"  20 features (C2/C4):  {feat20_ms:.2f} ms\n")

        print("Per-feature timing (marginal cost, shared intermediates pre-computed, avg over 5 runs):")
        print(f"  {'#':<4} {'Description':<30} {'ms':>8}")
        print(f"  {'-'*46}")
        feat_rows = time_features_individually(A, n_reps=5)
        for fid, desc, ms in feat_rows:
            print(f"  {fid:<4} {desc:<30} {ms:>8.3f}")
        print(f"  {'-'*46}")
        print(f"  {'Total (sum of marginal)':<34} {sum(ms for _,_,ms in feat_rows):>8.3f}\n")
    else:
        feat14_ms = sum(ms for fid, _, ms in feat_rows if int(fid) < 14)
        feat20_ms = sum(ms for _, _, ms in feat_rows)

    # Image rendering (reuse if pre-computed)
    if render_64 is None:
        print("Image rendering times (avg over 5 runs):")
        print(f"  {'Mode':<28} {'64px (ms)':>10} {'128px (ms)':>11}")
        print(f"  {'-'*52}")

        render_64  = {}
        render_128 = {}
        total_64 = total_128 = 0.0

        for mode in IMAGE_MODES:
            t0 = time.perf_counter()
            for _ in range(5):
                sparsity_image(A, size=64, mode=mode)
            ms_64 = (time.perf_counter() - t0) / 5 * 1000

            t0 = time.perf_counter()
            for _ in range(5):
                sparsity_image(A, size=128, mode=mode)
            ms_128 = (time.perf_counter() - t0) / 5 * 1000

            render_64[mode]  = ms_64
            render_128[mode] = ms_128
            total_64  += ms_64
            total_128 += ms_128
            print(f"  {mode:<28} {ms_64:>10.2f} {ms_128:>11.2f}")

        print(f"  {'-'*52}")
        print(f"  {'Total (all 14 modes)':<28} {total_64:>10.2f} {total_128:>11.2f}\n")

    # Pipeline summary
    feat_ms  = feat14_ms if model["feat_n"] == 14 else feat20_ms
    renders  = render_64 if model["size"] == 64 else render_128
    img_ms   = sum(renders[m] for m in model["modes"])
    pipe_ms  = feat_ms + img_ms + model["infer_ms"]
    total_ms = pipe_ms + best_ms * model["mrt"]

    print(f"\nPipeline summary — {model['label']}:")
    print(f"  Feature extraction ({model['feat_n']}):  {feat_ms:.2f} ms")
    for m in model["modes"]:
        print(f"  {m} ({model['size']}px):  {renders[m]:.2f} ms")
    print(f"  Model inference ({model['size']}px):     {model['infer_ms']:.1f} ms")
    print(f"  {'─'*38}")
    print(f"  Pipeline overhead:             {pipe_ms:.2f} ms")
    print(f"  Predicted solve (best×{model['mrt']}):  {best_ms * model['mrt']:.2f} ms")
    print(f"  Total (pipeline + solve):      {total_ms:.2f} ms")

    # Random baseline
    # Every solver is killed after KILL_MS if it has not yet converged.
    # Success: converged before KILL_MS → cost = actual elapsed time.
    # Failure (DNF or killed): cost = time until failure detected (≤ KILL_MS)
    #                                 + mean of all successful solvers
    #   (after wasting time on a failure the system still needs solving,
    #    so the expected cost of retrying is added on top — no further cap).
    # Random baseline — geometric retry model
    # Any solver is killed after KILL_MS. Solvers that converge before that are
    # "successes"; the rest are "failures". When picking randomly until a solver
    # succeeds the expected total cost is:
    #
    #   E = mean_success + (n_failed / n_success) × mean_detection_time
    #
    # The (n_failed/n_success) term is the expected number of failed attempts
    # before drawing a successful solver (geometric distribution, exact result).
    KILL_MS = 10_000.0

    success_ms   = [ms for _, ms, c in solver_results if c and ms < KILL_MS]
    fail_detect  = [min(ms, KILL_MS) for _, ms, c in solver_results if not c or ms >= KILL_MS]
    n_success    = len(success_ms)
    n_failed     = len(fail_detect)
    mean_succ_ms = float(np.mean(success_ms)) if success_ms else KILL_MS
    mean_det_ms  = float(np.mean(fail_detect)) if fail_detect else 0.0
    random_exp_ms = mean_succ_ms + (n_failed / n_success) * mean_det_ms if n_success else KILL_MS

    print(f"\n  Random baseline — geometric retry model (kill at {KILL_MS:.0f} ms):")
    print(f"    Successful (<10 s): {n_success}/19,  mean = {mean_succ_ms:.2f} ms")
    print(f"    Failed/killed:      {n_failed}/19,  mean detection = {mean_det_ms:.2f} ms")
    print(f"    E = {mean_succ_ms:.2f} + ({n_failed}/{n_success}) × {mean_det_ms:.2f} = {random_exp_ms:.2f} ms")
    print(f"    Expected random total:        {random_exp_ms:.2f} ms")
    print(f"\n  Speedup of model over random:  {random_exp_ms / total_ms:.2f}×")
    print(f"  Pipeline as % of model total:  {pipe_ms / total_ms * 100:.1f}%")

    return dict(solver_results=solver_results, feat_rows=feat_rows,
                render_64=render_64, render_128=render_128)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    n_50k  = GRID_SIZE_50K  ** 2   # 50,176
    n_100k = GRID_SIZE_100K ** 2   # 100,489

    matrices = [
        ("SPD Poisson 2D  ~50k",   poisson_2d(GRID_SIZE_50K)),
        ("SPD Poisson 2D  ~100k",  poisson_2d(GRID_SIZE_100K)),
        ("Nonsymmetric    ~50k",   random_nonsymmetric(n_50k,  NNZ_PER_ROW / n_50k,  rng)),
        ("Nonsymmetric    ~100k",  random_nonsymmetric(n_100k, NNZ_PER_ROW / n_100k, rng)),
    ]
    models = [MODEL_C3, MODEL_C4]

    for mat_label, A in matrices:
        # Run solvers, features, and rendering once; reuse for both model configs
        cache: dict = {}
        for model in models:
            result = run_matrix_benchmark(
                mat_label, A, model,
                solver_results = cache.get("solver_results"),
                feat_rows      = cache.get("feat_rows"),
                render_64      = cache.get("render_64"),
                render_128     = cache.get("render_128"),
            )
            if result:
                cache.update(result)


if __name__ == "__main__":
    main()
