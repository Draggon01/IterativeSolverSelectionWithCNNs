# combinedDatawithnewFeatures

Results from a full rerun of all 14 single-channel image modes at 64px,
with an expanded feature set (20 features) and convergence penalty enabled.

## What changed vs combinedDataFolder

### 1. Feature set: 14 → 20 features

Six new matrix features added to `model.py`'s `matrix_features()`:

| # | Name | Description | Motivation |
|---|------|-------------|------------|
| 14 | `neg_offdiag_frac` | Fraction of off-diagonal entries that are negative | M-matrices (SPD/FEM) have ~1.0; indefinite matrices ~0.5 — helps separate CG-type from GMRES-type |
| 15 | `pos_diag_frac` | Fraction of strictly positive diagonal entries | 1.0 = SPD-like; <1 = indefinite or singular-like — predicts symmlq/cr inapplicability |
| 16 | `struct_sym_frac` | Fraction of (i,j) entries with a matching (j,i) | Distinguishes structurally symmetric from truly asymmetric — helps gmres vs fgmres separation |
| 17 | `min_diag_dominance` | min row-wise \|d_i\| / Σ\|off_i\| | Detects single worst-case row for ILU breakdown — predicts fbcgsr+ilu / cg+ilu failure |
| 18 | `nnz_per_row_cv` | CV of per-row non-zero counts | Low = structured grid (GAMG works well); high = unstructured (GAMG less reliable) |
| 19 | `diag_dom_variance` | Variance of per-row dominance ratios | Patchy dominance (high variance) predicts ILU instability even when mean dominance is fine |

Old checkpoints (trained with 14 features) are **not compatible** with the new model.
All experiments must be retrained from scratch.

### 2. Convergence penalty enabled (λ = 1.0)

Loss function: `CE + 1.0 × mean(Σ softmax(logits)[non-converging solvers])`

Penalises the model for assigning probability mass to solvers that did not
converge for a given matrix. Addresses the high failure rate observed in
gmres+gamg (75%) and symmlq+icc (62%) in previous runs.

### 3. All 14 image modes, size 64px only

Previous grid searches covered 64+128px. This run fixes size=64 for speed
and covers all available modes including the rcm_* variants.

## Rerun procedure

```bash
cd experiments/thesis_experiments

# Step 1 — recompute features in the base dataset (no re-solving needed)
DATA_DIR=./data/base CACHE_DIR=../shared/cache/ \
docker compose run --rm recompute_features

# Step 2 — delete old incompatible checkpoints and stale multimode HDF5
rm -rf checkpoints/
rm -f data/multimode/dataset.h5

# Step 3 — retrain all 14 modes at 64px
SKIP_DATAGEN=1 \
CACHE_DIR=../shared/cache/ \
RESULTS_FILE=./experimentResults/combinedDataWithNewFeatures/newFeaturessummary.txt \
MODES="binary density log_density magnitude symmetry diagonal sign signed_magnitude \
       rcm_binary rcm_density rcm_log_density rcm_magnitude rcm_sign rcm_signed_magnitude" \
SIZES="64" \
CONVERGENCE_PENALTY=1.0 \
BATCH_SIZE=128 \
MODEL_SIZE=large \
./run_experiments.sh
```

## Single-channel results (large model, 20 features, λ=1.0, 64px)

Sorted by Acc% descending.

| Experiment               | Acc%  | MP%   | MR%   | F1%   | Fail% |
|--------------------------|-------|-------|-------|-------|-------|
| magnitude_64             | 64.29 | 66.58 | 66.36 | 63.38 | 4.23  |
| rcm_magnitude_64         | 63.47 | 66.80 | 65.74 | 62.79 | —     |
| symmetry_64              | 63.36 | 64.74 | 65.49 | 61.94 | 3.72  |
| rcm_log_density_64       | 62.75 | 66.52 | 65.27 | 62.12 | —     |
| rcm_signed_magnitude_64  | 62.54 | 65.17 | 64.09 | 61.36 | —     |
| log_density_64           | 62.23 | 66.12 | 65.13 | 61.65 | 4.54  |
| rcm_density_64           | 61.51 | 65.59 | 64.19 | 61.00 | —     |
| rcm_sign_64              | 61.51 | 65.43 | 63.82 | 60.87 | —     |
| signed_magnitude_64      | 61.20 | 63.39 | 62.78 | 59.85 | 5.16  |
| sign_64                  | 60.99 | 64.54 | 63.49 | 60.19 | 4.44  |
| nocnn                    | 59.75 | 62.40 | 62.72 | 59.08 | 4.95  |
| binary_64                | 59.34 | 62.66 | 62.53 | 58.87 | 5.06  |
| diagonal_64              | 59.13 | 62.96 | 62.11 | 58.46 | 5.16  |
| rcm_binary_64            | 59.03 | 62.82 | 62.04 | 58.30 | 5.99  |
| density_64               | 52.63 | 55.76 | 54.56 | 50.74 | 5.16  |

**Notes:** density_64 collapsed (~10pp drop vs previous run — likely convergence penalty interaction).
Failure rates much lower than previous run (~4-6% vs 20-75%). nocnn improved +1.44pp showing new features help the MLP branch.

**Top 6 by F1:** magnitude, rcm_magnitude, rcm_log_density, symmetry, rcm_signed_magnitude, log_density

## Dual-channel run command

15 pairwise combinations of the top 6 modes, large model, λ=1.0, at 64px and 128px:

```bash
SKIP_DATAGEN=1 \
CACHE_DIR=../shared/cache/ \
RESULTS_FILE=./experimentResults/combinedDataWithNewFeatures/newFeaturessummary.txt \
MODES="" SIZES="" \
DUAL_MODES="magnitude+rcm_magnitude \
magnitude+symmetry \
magnitude+rcm_signed_magnitude \
magnitude+rcm_log_density \
magnitude+log_density \
rcm_magnitude+symmetry \
rcm_magnitude+rcm_signed_magnitude \
rcm_magnitude+rcm_log_density \
rcm_magnitude+log_density \
symmetry+rcm_signed_magnitude \
symmetry+rcm_log_density \
symmetry+log_density \
rcm_signed_magnitude+rcm_log_density \
rcm_signed_magnitude+log_density \
rcm_log_density+log_density" \
DUAL_SIZES="64 128" \
MODEL_SIZE=large \
CONVERGENCE_PENALTY=1.0 \
BATCH_SIZE=64 \
./run_experiments.sh
```

Results append automatically to `newFeaturessummary.txt` below the single-channel results.

## Dual-channel results (large model, 20 features, λ=1.0, 64px)

Sorted by F1% descending. Best combo: **magnitude+log_density** (65.02% F1).

| Experiment                               | Acc%  | MP%   | MR%   | F1%   |
|------------------------------------------|-------|-------|-------|-------|
| magnitude__log_density_64                | 65.94 | 68.42 | 67.73 | 65.02 |
| magnitude__rcm_log_density_64            | 65.43 | 67.72 | 67.75 | 64.74 |
| magnitude__rcm_signed_magnitude_64       | 65.53 | 67.52 | 67.36 | 64.30 |
| magnitude__symmetry_64                   | 64.71 | 66.34 | 66.35 | 63.49 |
| rcm_magnitude__rcm_log_density_64        | 63.98 | 66.95 | 65.98 | 63.51 |
| rcm_magnitude__log_density_64            | 64.09 | 67.24 | 66.56 | 63.77 |
| rcm_magnitude__rcm_signed_magnitude_64   | 64.29 | 67.98 | 66.25 | 63.50 |
| symmetry__log_density_64                 | 64.50 | 66.70 | 66.88 | 63.57 |
| symmetry__rcm_log_density_64             | 64.29 | 67.12 | 66.69 | 63.43 |
| rcm_signed_magnitude__rcm_log_density_64 | 64.40 | 66.05 | 65.84 | 63.42 |
| magnitude__rcm_magnitude_64              | 63.78 | 66.49 | 65.76 | 63.37 |
| symmetry__rcm_signed_magnitude_64        | 64.09 | 65.96 | 65.58 | 62.62 |
| rcm_magnitude__symmetry_64               | 63.67 | 64.68 | 65.37 | 62.66 |
| rcm_signed_magnitude__log_density_64     | 63.47 | 66.25 | 65.69 | 62.40 |
| rcm_log_density__log_density_64          | 62.02 | 66.30 | 65.07 | 61.44 |

## Dual-channel results (large model, 20 features, λ=1.0, 128px)

Sorted by F1% descending. Best combo: **magnitude+log_density** (65.60% F1), best Acc: **magnitude+rcm_log_density** (66.56%).

| Experiment                                | Acc%  | MP%   | MR%   | F1%   |
|-------------------------------------------|-------|-------|-------|-------|
| magnitude__log_density_128                | 65.94 | 68.67 | 68.22 | 65.60 |
| magnitude__rcm_log_density_128            | 66.56 | 69.15 | 68.88 | 65.53 |
| magnitude__rcm_signed_magnitude_128       | 65.84 | 69.39 | 67.59 | 65.16 |
| magnitude__rcm_magnitude_128              | 65.02 | 66.73 | 66.62 | 64.22 |
| rcm_magnitude__symmetry_128               | 65.02 | 66.05 | 66.83 | 64.23 |
| rcm_magnitude__log_density_128            | 64.60 | 67.97 | 67.00 | 64.35 |
| rcm_magnitude__rcm_signed_magnitude_128   | 64.60 | 67.94 | 66.77 | 64.30 |
| rcm_magnitude__rcm_log_density_128        | 64.50 | 68.06 | 67.03 | 64.19 |
| symmetry__log_density_128                 | 65.12 | 66.68 | 67.14 | 64.10 |
| magnitude__symmetry_128                   | 64.60 | 66.86 | 66.41 | 63.87 |
| symmetry__rcm_log_density_128             | 64.71 | 66.92 | 66.98 | 63.84 |
| symmetry__rcm_signed_magnitude_128        | 64.09 | 65.31 | 66.05 | 62.89 |
| rcm_signed_magnitude__log_density_128     | 63.36 | 66.43 | 65.51 | 62.54 |
| rcm_signed_magnitude__rcm_log_density_128 | 63.67 | 66.50 | 65.11 | 62.52 |
| rcm_log_density__log_density_128          | 63.26 | 66.32 | 65.71 | 62.28 |

**Key findings:**
- `magnitude` is the indispensable channel — every top combination includes it; without it results drop to 62–63%
- 128px consistently beats 64px by +0.5–1.3pp
- Best second channel: `log_density` (best F1) or `rcm_log_density` (best Acc) — both outperform `symmetry` and `rcm_signed_magnitude`
- RCM reordering hurts magnitude (rcm_magnitude+anything < magnitude+same) but helps density/log_density
- Failure rates uniformly 3–5% — convergence penalty is effective
- Dual > single: best dual (66.56%) beats best single (64.29%) by +2.3pp

**Suggested ensemble:** `magnitude__rcm_log_density_128` + `magnitude__log_density_128` + `magnitude__rcm_signed_magnitude_128` — complementary second channels, expected to push past the old 67.39% ensemble.

## Ensemble run command (Campaign 4)

Top 3 by F1 at 128px, with complementary second channels. Add `rcm_magnitude__log_density_128`
as a 4th member to diversify the first channel:

```bash
rm data/multimode/dataset.h5 && \
IMAGE_SIZE=128 \
EXPERIMENTS="magnitude__rcm_log_density_128 magnitude__log_density_128 magnitude__rcm_signed_magnitude_128 rcm_magnitude__log_density_128" \
docker compose run --rm ensemble_evaluate
```

Run from `experiments/thesis_experiments/`. The `rm` is required — without it the script
sees all modes already present and skips re-rendering, even if `IMAGE_SIZE` changed.
`IMAGE_SIZE=128` must be set explicitly; the default is 64 which silently produces wrong results
(AdaptiveAvgPool handles arbitrary input sizes so no error is raised).

## Ensemble results (Campaign 4)

| Members | Acc% | MP% | MR% | F1% | Top-2% | Top-3% | Fail% | MRT× | ms/sample |
|---------|------|-----|-----|-----|--------|--------|-------|------|-----------|
| magnitude__rcm_log_density_128 + magnitude__log_density_128 + magnitude__rcm_signed_magnitude_128 + rcm_magnitude__log_density_128 | 66.98 | 69.78 | 69.20 | 66.51 | 87.20 | 94.53 | 3.72 | 1.216 | 50.9 |
| magnitude__rcm_log_density_128 + magnitude__rcm_signed_magnitude_128 + rcm_magnitude__symmetry_128 + symmetry__log_density_128 | 66.25 | 68.30 | 68.47 | 65.37 | 87.51 | 94.84 | 3.41 | 3.033 | 50.4 |
| magnitude__log_density_128 + magnitude__rcm_log_density_128 + rcm_magnitude__rcm_signed_magnitude_128 + symmetry__log_density_128 | 67.29 | 70.37 | 69.48 | 66.77 | 87.41 | 94.94 | 3.92 | 1.043 | 46.5 |

Gain over best individual (magnitude__rcm_log_density_128, ensemble 1): +0.42pp Acc, +0.98pp F1.

Note: Campaign 3 ensemble (67.39% Acc, 66.37% F1, ~12ms) still leads in accuracy despite
using only 14 features and 64px. Campaign 4 ensemble leads in MP (69.78% vs 69.47%),
MR (69.20% vs 69.58%), and F1 (66.51% vs 66.37%) but at 4× the inference cost (50.9ms vs 12ms).

Notable per-class results:
- cr+eisenstat: 89.1% F1 (up from 85.5% in Campaign 3) — convergence penalty helps
- symmlq+sor: 84.6% F1 (up from 77.8%) — big improvement
- cg+ilu: 49.1% F1, 31.5% Fail% — worse than Campaign 3 (56.9% F1, 11.9% Fail%)
- gmres+gamg: 20.6% F1 — unchanged, still the hardest class

## Diverse ensemble run command (Campaign 4, attempt 2)

3 different first channels (magnitude × 2, rcm_magnitude, symmetry), all 4 second channels different:

```bash
rm data/multimode/dataset.h5 && \
IMAGE_SIZE=128 \
EXPERIMENTS="magnitude__rcm_log_density_128 magnitude__rcm_signed_magnitude_128 rcm_magnitude__symmetry_128 symmetry__log_density_128" \
docker compose run --rm ensemble_evaluate
```

Note: `rm` required — multimode HDF5 needs `images_symmetry` which wasn't rendered before.

## Diverse ensemble run command (Campaign 4, attempt 3)

2 magnitude + 1 rcm_magnitude + 1 symmetry, all different second channels:

```bash
rm data/multimode/dataset.h5 && \
IMAGE_SIZE=128 \
EXPERIMENTS="magnitude__log_density_128 magnitude__rcm_log_density_128 rcm_magnitude__rcm_signed_magnitude_128 symmetry__log_density_128" \
docker compose run --rm ensemble_evaluate
```

## 5-model ensemble run command (Campaign 4, attempt 4)

Top 3 individual models by F1 + best non-magnitude first channel (rcm_magnitude__symmetry) +
best symmetry-first model (symmetry__log_density). All 6 modes already in multimode HDF5
from attempt 3 — no `rm` needed (ensemble_evaluate.py auto-detects size mismatch).

Members:
- magnitude__log_density_128        (65.60% F1, best individual)
- magnitude__rcm_log_density_128    (65.53% F1, best individual Acc: 66.56%)
- magnitude__rcm_signed_magnitude_128 (65.16% F1, #3 individual)
- rcm_magnitude__symmetry_128       (64.23% F1, best non-magnitude first channel)
- symmetry__log_density_128         (64.10% F1, diverse first channel)

First channels: magnitude×3, rcm_magnitude×1, symmetry×1
Second channels: log_density×2, rcm_log_density, rcm_signed_magnitude, symmetry

```bash
IMAGE_SIZE=128 \
EXPERIMENTS="magnitude__log_density_128 magnitude__rcm_log_density_128 magnitude__rcm_signed_magnitude_128 rcm_magnitude__symmetry_128 symmetry__log_density_128" \
docker compose run --rm ensemble_evaluate
```

Results: **worse than attempt 3** in all metrics — weaker members dilute the strong magnitude trio.
Attempt 3 remains the best Campaign 4 ensemble. Not used in the thesis.

| Members | Acc% | MP% | MR% | F1% | Top-2% | Top-3% | Fail% | MRT× | ms/sample |
|---------|------|-----|-----|-----|--------|--------|-------|------|-----------|
| magnitude__log_density + magnitude__rcm_log_density + magnitude__rcm_signed_magnitude + rcm_magnitude__symmetry + symmetry__log_density (128px, 5 models) | 66.46 | 68.82 | 68.75 | 65.92 | 87.51 | 95.15 | 3.92 | 1.206 | 59.5 |

## Reference: previous best results (14 features, no penalty, small model, 64px)

| Mode                  | Acc%  | F1%   |
|-----------------------|-------|-------|
| magnitude             | 64.19 | 63.42 |
| rcm_signed_magnitude  | 63.67 | 62.24 |
| rcm_magnitude         | 63.05 | 62.55 |
| symmetry              | 62.95 | 61.85 |
| signed_magnitude      | 62.85 | 61.68 |
| sign                  | 60.37 | 59.29 |
| nocnn baseline        | 58.31 | 57.57 |
