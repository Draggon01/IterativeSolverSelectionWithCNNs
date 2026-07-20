# otherSolvers

Experiments using an alternative set of 15 solver--preconditioner pairs (set B),
selected to cover practical workhorse combinations not included in the original
MM-AutoSolver set. Controlled via `SOLVER_SET=alt` in all pipeline steps.

## Solver set B (15 pairs)

| # | Pair | Applicability |
|---|------|---------------|
| 0 | cg+gamg | SPD only |
| 1 | cg+none | SPD only |
| 2 | gmres+ilu | General |
| 3 | gmres+bjacobi | General |
| 4 | gmres+none | General |
| 5 | fgmres+ilu | General |
| 6 | fgmres+bjacobi | General |
| 7 | bcgs+ilu | General |
| 8 | bcgs+jacobi | General |
| 9 | bcgs+gamg | General |
| 10 | bcgs+none | General |
| 11 | tfqmr+ilu | General |
| 12 | tfqmr+jacobi | General |
| 13 | lgmres+ilu | General |
| 14 | gcr+ilu | General |

## Dataset generation

```bash
cd experiments/thesis_experiments

# Step 1 — generate synthetic data (at least 250 samples per class)
SOLVER_SET=alt \
N_SAMPLES=20000 \
MIN_PER_CLASS=250 \
STORE_MATRIX=1 \
SEED=42 \
DATA_DIR=./data/alt_solvers \
docker compose run -d --rm datagen

# Step 2 — add SuiteSparse benchmark matrices
SOLVER_SET=alt \
DATA_DIR=./data/alt_solvers \
CACHE_DIR=../shared/cache/ \
docker compose run --rm githubdata_ingest
```

**Note:** `lgmres` and `gcr` are less common winners — if MIN_PER_CLASS=250 causes
the run to take too long, fall back to MIN_PER_CLASS=150 and N_SAMPLES=30000.

## Training

```bash
SOLVER_SET=alt \
SKIP_DATAGEN=1 \
DATA_DIR=./data/alt_solvers \
CACHE_DIR=../shared/cache/ \
RESULTS_FILE=./experimentResults/otherSolvers/otherSolvers_summary.txt \
MODES="magnitude log_density rcm_magnitude rcm_log_density symmetry" \
SIZES="64" \
CONVERGENCE_PENALTY=1.0 \
BATCH_SIZE=128 \
MODEL_SIZE=large \
./run_experiments.sh
```

## Results

<!-- Fill in after training completes -->

| Experiment | Acc% | MP% | MR% | F1% | Fail% |
|------------|------|-----|-----|-----|-------|
| magnitude_64 | ? | ? | ? | ? | ? |
| log_density_64 | ? | ? | ? | ? | ? |
| rcm_magnitude_64 | ? | ? | ? | ? | ? |
| rcm_log_density_64 | ? | ? | ? | ? | ? |
| symmetry_64 | ? | ? | ? | ? | ? |
| nocnn | ? | ? | ? | ? | ? |
