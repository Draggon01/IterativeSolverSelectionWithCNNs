"""
benchmark_solvers.py — Run every applicable PETSc Krylov solver on a given
sparse system and report detailed results.

Used to validate model predictions against ground-truth solver performance
and to generate per-matrix solver rankings for analysis.

Environment variables:
  MATRIX_PATH    scipy .npz file to load (optional; random matrix used if unset)
  BENCHMARK_T    Per-solver time limit in seconds   (default 30)
  TOL            Convergence tolerance              (default 1e-8)
  MAX_ITER       Hard iteration cap per solver      (default 5000)
  RESULTS_PATH   JSON file to write results to      (optional)
  DEVICE         Ignored here (CPU-only PETSc run)

Usage examples (inside the Docker predict service):
  docker compose run --rm predict python benchmark_solvers.py
  MATRIX_PATH=/workspace/data/my_matrix.npz BENCHMARK_T=60 \\
      docker compose run --rm predict python benchmark_solvers.py
"""

import os
import json
import time
import logging
import concurrent.futures
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from model import SOLVERS, SOLVER_IDX, matrix_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MATRIX_PATH  = os.getenv("MATRIX_PATH",   "")
BENCHMARK_T  = float(os.getenv("BENCHMARK_T", "30"))
TOL          = float(os.getenv("TOL",         "1e-8"))
MAX_ITER     = int(os.getenv("MAX_ITER",      "5000"))
RESULTS_PATH = os.getenv("RESULTS_PATH",  "")


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    solver:    str
    converged: bool
    iterations: int
    wall_time:  float     # seconds; inf if timed-out or errored
    residual:   float     # final relative residual norm


# ── PETSc solve with thread-based timeout ─────────────────────────────────────

def _csr_to_petsc(A: sp.csr_matrix) -> PETSc.Mat:
    n   = A.shape[0]
    mat = PETSc.Mat().createAIJWithArrays(
        (n, n),
        (A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.copy()),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    return mat


def _solve(A: sp.csr_matrix, b: np.ndarray, ksp_type: str) -> SolverResult:
    """Blocking PETSc solve; called inside a worker thread for timeout support."""
    mat = _csr_to_petsc(A)
    x   = mat.createVecRight()
    rhs = mat.createVecLeft()
    rhs.setValues(np.arange(len(b), dtype=np.int32), b.astype(np.float64))
    rhs.assemble()

    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=TOL, atol=1e-50, divtol=1e5, max_it=MAX_ITER)
    ksp.setConvergenceHistory()

    t0 = time.perf_counter()
    try:
        ksp.solve(rhs, x)
        elapsed   = time.perf_counter() - t0
        converged = ksp.getConvergedReason() > 0
        iters     = ksp.getIterationNumber()

        # Compute true relative residual ‖b − Ax‖ / ‖b‖
        x_arr = x.getArray().copy()
        res   = float(np.linalg.norm(b - A @ x_arr) / (np.linalg.norm(b) + 1e-12))
    except Exception as exc:
        log.debug("KSP %s raised: %s", ksp_type, exc)
        elapsed, converged, iters, res = float("inf"), False, -1, float("inf")
    finally:
        ksp.destroy()
        mat.destroy()
        x.destroy()
        rhs.destroy()

    return SolverResult(ksp_type, converged, iters, elapsed, res)


def run_solver_timed(
    A: sp.csr_matrix,
    b: np.ndarray,
    ksp_type: str,
    timeout: float = BENCHMARK_T,
) -> SolverResult:
    """Run _solve in a thread; return a timed-out result if it exceeds timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_solve, A, b, ksp_type)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log.warning("  %s timed out after %.1fs", ksp_type, timeout)
            return SolverResult(ksp_type, False, -1, float("inf"), float("inf"))


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(A: sp.csr_matrix, b: np.ndarray) -> list[SolverResult]:
    """Run all solvers and return results sorted by wall time (converged first)."""
    PETSc.Options()["ksp_error_if_not_converged"] = False

    results: list[SolverResult] = []
    for solver in SOLVERS:
        log.info("Running %-8s ...", solver)
        r = run_solver_timed(A, b, solver)
        results.append(r)
        status = f"converged in {r.iterations} iters  t={r.wall_time:.3f}s  res={r.residual:.2e}" \
                 if r.converged else "DID NOT CONVERGE"
        log.info("  %-8s  %s", solver, status)

    # Sort: converged solvers by time, then non-converged
    return sorted(results, key=lambda r: (not r.converged, r.wall_time))


def print_table(results: list[SolverResult], A: sp.csr_matrix) -> None:
    feats = matrix_features(A)
    print(f"\nMatrix: shape={A.shape}  nnz={A.nnz}  "
          f"density={feats[2]:.4f}  symmetry={feats[3]:.4f}")
    print(f"\n{'Solver':<10} {'Converged':<11} {'Iterations':<12} "
          f"{'Time (s)':<12} {'Residual':<12} {'Rank'}")
    print("-" * 65)
    rank = 1
    for r in results:
        conv_str = "YES" if r.converged else "NO"
        t_str    = f"{r.wall_time:.4f}" if r.wall_time < 1e9 else "timeout"
        res_str  = f"{r.residual:.2e}"  if r.residual  < 1e9 else "—"
        rank_str = str(rank) if r.converged else "—"
        print(f"{r.solver:<10} {conv_str:<11} {r.iterations:<12} "
              f"{t_str:<12} {res_str:<12} {rank_str}")
        if r.converged:
            rank += 1
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if MATRIX_PATH and os.path.exists(MATRIX_PATH):
        A = sp.load_npz(MATRIX_PATH).tocsr().astype(np.float64)
        log.info("Loaded matrix from %s  shape=%s  nnz=%d", MATRIX_PATH, A.shape, A.nnz)
    else:
        from generate_data import sample_matrix
        rng = np.random.default_rng(0)
        A, mat_type = sample_matrix(rng)
        log.info("Using random example  type=%s  shape=%s  nnz=%d",
                 mat_type, A.shape, A.nnz)

    n = A.shape[0]
    b = np.random.default_rng(1).standard_normal(n)

    log.info("Benchmarking all solvers (timeout=%.1fs per solver) ...", BENCHMARK_T)
    results = benchmark(A, b)
    print_table(results, A)

    if RESULTS_PATH:
        payload = {
            "matrix": {"shape": list(A.shape), "nnz": A.nnz},
            "results": [asdict(r) for r in results],
        }
        with open(RESULTS_PATH, "w") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Results written to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
