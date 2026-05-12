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
| `03_feature_distributions.png` | All 8 features split by best solver |
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
| `images` | (N, 64, 64) | float32 | Binary sparsity-pattern image |
| `features` | (N, 8) | float32 | Scalar matrix statistics |
| `labels` | (N,) | int32 | Index of best solver |
| `top3_labels` | (N, 3) | int8 | Top-3 solver indices ranked by time; -1 = no rank |
| `runtimes` | (N, N_SOLVERS) | float32 | Per-solver wall time in seconds; NaN = no data |
| `source` | (N,) | str | Origin, e.g. `synthetic/poisson2d`, `suitesparse/HB/bcsstk01` |

Attribute `solvers` on the root group gives the ordered solver name list.

---

## References

See `thesis/workdir/bibliography.bib`.
