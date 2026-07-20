# MM-AutoSolver Baseline — Results

Re-implementation of Xiong et al. (2025), "MM-AutoSolver: A multimodal machine learning method
for the auto-selection of iterative solvers and preconditioners",
*Journal of Parallel and Distributed Computing* 205, 105144.

---

## Overall Metrics

| Metric | Ours | Paper |
|---|---|---|
| Accuracy | **81.97%** | 78.54% |
| Macro Precision | 62.88% | 63.41% |
| Macro Recall | 61.83% | 62.81% |
| Macro F1 | 60.45% | 62.53% |
| Top-2 Accuracy | 90.13% | — |
| Top-3 Accuracy | 95.01% | — |

Additional metric not reported in the paper:

| Near-optimal Accuracy | Score |
|---|---|
| Within ±5% of best runtime | 86.97% |
| Within ±10% of best runtime | 91.47% |
| Within ±20% of best runtime | 95.86% |

---

## Per-Class Results

Dataset: 8,214 samples, 10% held-out validation (821 samples).
Paper dataset: 11,623 samples (real SuiteSparse matrices).

| Solver | Ours (N) | Paper (N) | F1 | Precision | Recall |
|---|---|---|---|---|---|
| fbcgsr+jacobi | 2000 | 2173 | 98.8% | 98.6% | 99.0% |
| bcgsl+none | 2000 | 2054 | 97.3% | 96.6% | 98.0% |
| symmlq+icc | 161 | 1201 | 51.4% | 42.9% | 64.3% |
| symmlq+jacobi | 218 | 923 | 95.8% | 100.0% | 92.0% |
| dgmres+none | 64 | 650 | 66.7% | 60.0% | 75.0% |
| gmres+gamg | 395 | 640 | 41.0% | 38.1% | 44.4% |
| cr+eisenstat | 662 | 598 | 77.9% | 69.9% | 87.9% |
| symmlq+sor | 120 | 582 | 93.3% | 100.0% | 87.5% |
| fbcgsr+ilu | 622 | 562 | 93.2% | 100.0% | 87.3% |
| minres+gamg | 416 | 524 | 56.6% | 43.5% | 81.1% |
| fcg+gamg | 142 | 342 | 0.0% | 0.0% | 0.0% |
| cr+jacobi | 343 | 310 | 94.9% | 93.3% | 96.6% |
| cg+ilu | 304 | 275 | 28.6% | 46.7% | 20.6% |
| fgmres+gamg | 144 | 226 | 0.0% | 0.0% | 0.0% |
| cg+eisenstat | 248 | 224 | 52.5% | 53.3% | 51.6% |
| cg+bjacobi | 214 | 193 | 56.0% | 51.9% | 60.9% |
| cr+ilu | 75 | 68 | 0.0% | 0.0% | 0.0% |
| cgs+gamg | 54 | 49 | 44.4% | 100.0% | 28.6% |
| bcgsl+asm | 32 | 29 | 100.0% | 100.0% | 100.0% |

---

## Training Setup

| Parameter | Value |
|---|---|
| Epochs | 256 |
| Batch size | 512 |
| Optimizer | Adam, lr = 1×10⁻³ |
| Loss | CrossEntropy |
| Validation split | 10% (fixed seed) |
| Device | CUDA |

---

## Dataset Differences

| Aspect | Ours | Paper |
|---|---|---|
| Source | Synthetic + SuiteSparse | Real SuiteSparse matrices |
| Total samples | 8,214 | 11,623 |
| Matrix sizes | n = 100 – 84,617 | up to >1,000,000 rows |
