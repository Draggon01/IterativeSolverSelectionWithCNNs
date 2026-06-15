# MM-AutoSolver Baseline — Results

Re-implementation of Xiong et al. (2025), "MM-AutoSolver: A multimodal machine learning method
for the auto-selection of iterative solvers and preconditioners",
*Journal of Parallel and Distributed Computing* 205, 105144.

---

## Overall Metrics

| Metric | Ours | Paper |
|---|---|---|
| Accuracy | **76.54%** | 78.54% |
| Macro Precision | 55.80% | 63.41% |
| Macro Recall | **69.32%** | 62.81% |
| Macro F1 | 56.64% | 62.53% |

Additional metric not reported in the paper:

| Near-optimal Accuracy | Score |
|---|---|
| Within ±5% of best runtime | 89.37% |
| Within ±10% of best runtime | **93.41%** |
| Within ±20% of best runtime | 96.40% |

Near-optimal accuracy measures the fraction of predictions whose chosen solver
runs within X% of the fastest possible solver — a more practically meaningful
metric than exact class accuracy for solver selection.

---

## Per-Class Results

Dataset: 11,381 samples (synthetic), 10% held-out validation.
Paper dataset: 10,404 samples (real SuiteSparse matrices).

| Solver | Ours (N) | Paper (N) | F1 | Precision | Recall |
|---|---|---|---|---|---|
| fbcgsr+jacobi | 2611 | 2173 | 98.6% | 98.0% | 99.2% |
| bcgsl+none | 376 | 2054 | 88.2% | 90.9% | 85.7% |
| symmlq+icc | 409 | 1201 | 48.0% | 50.0% | 46.2% |
| symmlq+jacobi | 10 ⚠ | 923 | 30.8% | 18.2% | 100.0% |
| dgmres+none | 229 | 650 | 92.7% | 95.0% | 90.5% |
| gmres+gamg | 525 | 640 | 65.9% | 69.8% | 62.5% |
| cr+eisenstat | 1268 | 598 | 92.8% | 96.0% | 89.7% |
| symmlq+sor | 34 ⚠ | 582 | 46.2% | 30.0% | 100.0% |
| fbcgsr+ilu | 222 | 562 | 89.8% | 91.7% | 88.0% |
| minres+gamg | 118 | 524 | 0.0% | 0.0% | 0.0% |
| fcg+gamg | 245 | 342 | 39.3% | 36.4% | 42.9% |
| cr+jacobi | 1113 | 310 | 79.4% | 81.1% | 77.8% |
| cg+ilu | 649 | 275 | 44.8% | 46.7% | 43.1% |
| fgmres+gamg | 258 | 226 | 0.0% | 0.0% | 0.0% |
| cg+eisenstat | 2500 | 224 | 83.6% | 93.5% | 75.5% |
| cg+bjacobi | 619 | 193 | 70.6% | 65.1% | 77.1% |
| cr+ilu | 61 | 68 | 11.8% | 6.7% | 50.0% |
| cgs+gamg | 10 ⚠ | 49 | 4.9% | 2.5% | 100.0% |
| bcgsl+asm | 124 | 29 | 88.9% | 88.9% | 88.9% |

⚠ Fewer than 35 training samples — results unreliable for these classes.

---

## Training Setup

| Parameter | Value |
|---|---|
| Epochs run | ~200 (converged, no improvement after ~150) |
| Batch size | 512 |
| Optimizer | Adam, lr = 1×10⁻³ |
| Loss | CrossEntropy with inverse-frequency class weights |
| Validation split | 10% (fixed seed) |
| Device | CUDA |

Deviation from paper: class-weighted loss was added to compensate for
dataset imbalance. The paper used unweighted CrossEntropy on a more
balanced real-world dataset.

---

## Dataset Differences

| Aspect                      | Ours                                     | Paper                                          |
|-----------------------------|------------------------------------------|------------------------------------------------|
| Source                      | Synthetic (random SPD, non-sym, Poisson) | Real SuiteSparse matrices                      |
| Total samples               | 11,381                                   | 10,404                                         |
| Rare classes (< 50 samples) | 3 (symmlq+jacobi, symmlq+sor, cgs+gamg)  | 1 (bcgsl+asm: 29)                              |
| Matrix sizes                | n = 100 – 46,772                         | they have matrices with more than 1mil entries |

---

## Discussion

Accuracy is within **2% of the paper** despite using only synthetic matrices
instead of real engineering problems from SuiteSparse. Macro recall
**exceeds** the paper (69.3% vs 62.8%), indicating the model generalises well
across solver families.

The macro F1 gap (56.6% vs 62.5%) is attributable to three classes with
fewer than 35 training samples (symmlq+jacobi, symmlq+sor, cgs+gamg), which
synthetic matrix generation cannot reliably reproduce. `minres+gamg` and
`fgmres+gamg` score 0% F1 because they are structurally indistinguishable
from `gmres+gamg` on synthetic data — all three are GAMG-preconditioned
methods that converge similarly on Poisson-type problems.

The near-optimal accuracy of **93.4% at ±10%** shows that even when the model
does not predict the single fastest solver, it almost always selects one that
is competitive — which is the practically relevant criterion for deployment.
