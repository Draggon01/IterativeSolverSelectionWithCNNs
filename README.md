# Investigation of Convolutional Neural Networks for Iterative Solver Selection

Bachelor's thesis — TUM Informatics, Leon Unterberger, 2026.

Trains a CNN+MLP classifier to predict the best PETSc Krylov solver for a
sparse linear system A x = b, given only the sparsity pattern and scalar
statistics of A.

---

## Repository layout

```
src/                        Python pipeline scripts
  model.py                  Shared model architecture + feature extraction
  generate_data.py          Generate synthetic training matrices
  ingest_suitesparse.py     Download and ingest SuiteSparse matrices
  train_solver_selector.py  Train the CNN classifier
  predict.py                Predict the best solver for a new matrix
  benchmark_solvers.py      Benchmark all solvers against each other
  visualize.py              Save static analysis figures (PNG)
  browse_data.py            Interactive matrix browser (requires display)

container/
  Dockerfile
  docker-compose.yml
  requirements.txt

thesis/workdir/             LaTeX thesis source (TUM template)
```

---

## Local setup

Requires Python 3.10+, a working PETSc installation, and a display for
the interactive browser.

```bash
# 1. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install PETSc first (petsc4py must link against it)
pip install "petsc>=3.20"
pip install --no-build-isolation "petsc4py>=3.20"

# 3. Install remaining dependencies
pip install scipy h5py tensorboard matplotlib ssgetpy

# 4. All pipeline scripts run from src/
cd src/
```

If `pip install petsc` fails with a BLAS error, install BLAS first:
```bash
sudo apt-get install libopenblas-dev   # Ubuntu/Debian
sudo dnf install openblas-devel        # Fedora/RHEL
brew install openblas                  # macOS
```

---

## Local pipeline (step by step)

### Step 1 — Generate synthetic training data

```bash
N_SAMPLES=1000 DATA_DIR=./data python generate_data.py
```

| Variable | Default | Description |
|---|---|---|
| `N_SAMPLES` | 1000 | Number of matrices to generate |
| `DATA_DIR` | `/workspace/data` | Output directory for `dataset.h5` |
| `MAX_ITER` | 2000 | Max KSP iterations per solver |
| `TOL` | 1e-8 | Convergence tolerance |
| `SEED` | 42 | RNG seed |

### Step 1b — Ingest SuiteSparse matrices (optional)

Appends real-world matrices from the SuiteSparse collection to the same
`dataset.h5`.  Requires `ssgetpy` (`pip install ssgetpy`).

```bash
# Auto mode — download directly from SuiteSparse
MODE=auto MIN_N=100 MAX_N=10000 N_MATRICES=200 DATA_DIR=./data \
    python ingest_suitesparse.py

# Manual mode — use locally downloaded .mtx files
MODE=manual MTX_DIR=./data/mtx DATA_DIR=./data \
    python ingest_suitesparse.py
```

| Variable | Default | Description |
|---|---|---|
| `MODE` | `auto` | `auto` (ssgetpy) or `manual` (local `.mtx` files) |
| `MIN_N` | 100 | Minimum matrix dimension |
| `MAX_N` | 50000 | Maximum matrix dimension |
| `N_MATRICES` | 200 | Maximum matrices to ingest |
| `ONLY_SPD` | 0 | Set to `1` to restrict to SPD matrices |
| `MTX_DIR` | `./data/mtx` | Directory of `.mtx` files (manual mode) |
| `CACHE_DIR` | `./data/suitesparse_cache` | Download cache (auto mode) |

Safe to re-run — already-ingested matrices are skipped automatically.

### Step 2 — Train the classifier

```bash
DATA_DIR=./data \
CHECKPOINT_DIR=./checkpoints \
LOG_DIR=./logs \
MAX_EPOCHS=100 \
BATCH_SIZE=256 \
LEARNING_RATE=0.0003 \
    python train_solver_selector.py
```

| Variable | Default | Description |
|---|---|---|
| `MAX_EPOCHS` | 100 | Total training epochs |
| `BATCH_SIZE` | 256 | Samples per forward/backward pass |
| `LEARNING_RATE` | 3e-4 | Initial AdamW learning rate |
| `CHECKPOINT_EVERY` | 5 | Save a checkpoint every N epochs |
| `KEEP_LAST_N` | 3 | Number of checkpoints to retain |
| `VAL_SPLIT` | 0.15 | Fraction of data used for validation |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

Training automatically resumes from the latest checkpoint if one exists.
Monitor live with TensorBoard:
```bash
tensorboard --logdir=./logs   # open http://localhost:6006
```

### Step 3 — Predict the best solver for a new matrix

```bash
CHECKPOINT_DIR=./checkpoints python predict.py

# Use a specific matrix (scipy .npz format)
MATRIX_PATH=./my_matrix.npz CHECKPOINT_DIR=./checkpoints python predict.py
```

### Step 4 — Benchmark all solvers (validation)

Runs every solver against a matrix and prints a ranked timing table.

```bash
python benchmark_solvers.py

# With a specific matrix and 60-second per-solver timeout
MATRIX_PATH=./my_matrix.npz BENCHMARK_T=60 python benchmark_solvers.py

# Save results to JSON
RESULTS_PATH=./results.json python benchmark_solvers.py
```

### Step 5 — Visualise the dataset

```bash
# Save static PNG figures to ./viz/
DATA_DIR=./data CHECKPOINT_DIR=./checkpoints VIZ_DIR=./viz python visualize.py

# Also open figures interactively
SHOW=1 DATA_DIR=./data python visualize.py
```

Produces four figures:

| File | Contents |
|---|---|
| `01_dataset_overview.png` | Label distribution, size/density histograms, box plots |
| `02_sparsity_gallery.png` | 24 example sparsity patterns coloured by best solver |
| `03_feature_distributions.png` | All 14 features split by best solver |
| `04_predictions.png` | Confusion matrix, per-solver accuracy, probability heatmap |

### Step 6 — Interactive matrix browser

```bash
DATA_DIR=./data CHECKPOINT_DIR=./checkpoints python browse_data.py
```

Navigate through individual matrices, see sparsity patterns, solver
probabilities, actual runtimes, and top-3 solver rankings.  Use the
**[View]** button to toggle the right panel between Features / Runtimes / Info.

Requires a display.  On a remote server use X11 forwarding:
```bash
ssh -X user@server
```

---

## Docker pipeline

Build the image once, then run each step with `docker compose run`:

```bash
cd container/

# Step 1 — generate synthetic data
docker compose run datagen

# Step 1b — ingest SuiteSparse (optional)
docker compose run ingest

# Step 2 — train
docker compose run trainer

# Step 3 — predict
docker compose run predict

# Step 4 — benchmark
docker compose run --rm predict python benchmark_solvers.py

# Step 5 — visualise (figures written to container/viz/)
docker compose run --rm predict python visualize.py

# Monitor training live
docker compose up tensorboard    # open http://localhost:6006
```

Enable GPU by uncommenting the `deploy.resources.reservations` block in
`docker-compose.yml` before running the trainer.

---

## Dataset format (`dataset.h5`)

| Dataset | Shape | dtype | Description |
|---|---|---|---|
| `images` | (N, 64, 64) | float32 | Sparsity-pattern image (encoding set by `IMAGE_MODE`) |
| `features` | (N, 14) | float32 | Scalar matrix statistics (see feature list below) |
| `labels` | (N,) | int32 | Index of best (KSP, PC) pair |
| `top3_labels` | (N, 3) | int8 | Top-3 pair indices ranked by wall time; -1 = no rank |
| `runtimes` | (N, N_SOLVERS) | float32 | Per-pair wall time in seconds; NaN = no data |
| `source` | (N,) | str | Origin, e.g. `synthetic/poisson2d`, `suitesparse/HB/bcsstk01` |

Root attributes: `solvers` (ordered list of 30 `ksp+pc` names), `image_mode` (encoding used).

### Feature vector (14 elements)

| # | Name | Description |
|---|---|---|
| 0 | `log(n)` | Log-scaled matrix dimension |
| 1 | `log(nnz)` | Log-scaled non-zero count |
| 2 | `density` | Fill ratio nnz / n² |
| 3 | `symmetry` | ‖A − Aᵀ‖_F / ‖A‖_F — 0 = symmetric |
| 4 | `diag dom.` | Mean \|diag\| / row sum (diagonal dominance) |
| 5 | `Frob/n` | Size-normalised Frobenius norm |
| 6 | `trace/n` | Size-normalised trace |
| 7 | `max/mean` | Max absolute entry / mean absolute entry |
| 8 | `spectral rad.` | log(1 + spectral radius estimate, power iteration) |
| 9 | `log cond.` | log(1 + max\|diag\| / min\|diag\|) — condition proxy |
| 10 | `bandwidth/n` | Max \|i − j\| over all non-zeros, normalised |
| 11 | `diag nnz frac` | Fraction of non-zero diagonal entries |
| 12 | `row norm CV` | Coefficient of variation of row norms |
| 13 | `offdiag Frob` | ‖A − D‖_F / ‖A‖_F — off-diagonal energy fraction |

---

## Experiments

Each experiment changes one variable, regenerates data, and trains a fresh model.
Compare runs using `val_acc` in TensorBoard (`tensorboard --logdir=./logs`).

### Experiment A — Image encoding

Controls how the sparse matrix is compressed into the 64×64 CNN input image.

| `IMAGE_MODE` | Pixel value | What the CNN sees |
|---|---|---|
| `binary` | 0 or 1 | Only position of non-zeros |
| `density` | count / block area | How many non-zeros per region |
| `log_density` | log(1 + count), normalised | Density with compressed dynamic range |
| `magnitude` | mean \|value\|, normalised | Magnitude of entries per region |

```bash
# Run all four — each writes to a separate data and checkpoint directory
for MODE in binary density log_density magnitude; do
  IMAGE_MODE=$MODE N_SAMPLES=10000 DATA_DIR=./data/$MODE python generate_data.py
  IMAGE_MODE=$MODE DATA_DIR=./data/$MODE CHECKPOINT_DIR=./checkpoints/$MODE \
    LOG_DIR=./logs/$MODE MAX_EPOCHS=200 python train_solver_selector.py
done

# Compare in TensorBoard
tensorboard --logdir=./logs   # open http://localhost:6006
```

Expected result: `log_density` and `density` outperform `binary` for large matrices
because the CNN can see how *dense* each region is, not just whether it is occupied.

---

### Experiment B — Feature ablation

Train on the full 14-feature vector, then retrain with reduced sets to see which
features matter most.  Edit `matrix_features()` in `model.py` to zero out or remove
individual features, or add new ones (e.g. exact eigenvalues via
`scipy.sparse.linalg.eigs`).

```bash
# Baseline — all 14 features
DATA_DIR=./data CHECKPOINT_DIR=./checkpoints/full_features \
  LOG_DIR=./logs/full_features python train_solver_selector.py

# Check which features correlate with label in the browser
DATA_DIR=./data python browse_data.py   # → [View] → Features panel
```

The features most likely to matter for solver selection:
- `spectral rad.` and `log cond.` — directly determine Krylov convergence rates
- `symmetry` — determines which KSP types are applicable
- `diag dom.` + `bandwidth/n` — determine how effective ILU/ICC preconditioners are

---

### Experiment C — Dataset composition

Compare models trained on different data sources.

```bash
# Synthetic only (fast, reproducible)
N_SAMPLES=50000 DATA_DIR=./data/synthetic python generate_data.py

# SuiteSparse only (real-world, download required)
MODE=auto N_MATRICES=500 DATA_DIR=./data/suitesparse python ingest_suitesparse.py

# Mixed — generate synthetic first, then append SuiteSparse
N_SAMPLES=50000 DATA_DIR=./data/mixed python generate_data.py
MODE=auto N_MATRICES=500 DATA_DIR=./data/mixed python ingest_suitesparse.py
```

Train and compare `val_acc` across the three datasets.  A model trained only on
synthetic data that generalises well to SuiteSparse matrices validates that the
synthetic generators capture the relevant structural variation.

---

### Experiment D — Solver + preconditioner label space

The label space is 30 `(KSP, PC)` pairs (6 solvers × 5 preconditioners).
To compare against the simpler 6-solver baseline, check out an earlier commit
before the preconditioner expansion, regenerate data, and train.

Alternatively, collapse predictions back to KSP type only at inference time
to measure how often the model picks the right *solver family* regardless of
which preconditioner it recommends:

```python
from model import SOLVER_PAIRS, SOLVER_NAMES
import numpy as np

# probs is the (30,) output of predict_solver()
ksp_probs = {}
for i, (ksp, pc) in enumerate(SOLVER_PAIRS):
    ksp_probs[ksp] = ksp_probs.get(ksp, 0.0) + probs[i]

best_ksp = max(ksp_probs, key=ksp_probs.get)
print("Best KSP family:", best_ksp)
print("Best pair:      ", SOLVER_NAMES[np.argmax(probs)])
```

---

## References

See `thesis/workdir/bibliography.bib`.
