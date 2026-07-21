# Investigation of Convolutional Neural Networks for Iterative Solver Selection

Bachelor's thesis — TUM Informatics, Leon Unterberger, 2026.

Trains a multimodal CNN+MLP classifier to predict the best PETSc Krylov solver and
preconditioner for a sparse linear system $Ax = b$, given the sparsity-pattern image and
scalar statistics of $A$.  The primary contribution is a systematic evaluation of 14 image
encoding strategies, dual-channel CNN inputs, and an ensemble approach.

---

## Repository layout

```
experiments/
  thesis_experiments/     Main CNN experiments (Campaigns 1–5)
    model.py              Architecture, 14/20 features, image rendering, 19 solver pairs
    generate.py           Synthetic data generation via PETSc benchmarking
    ingest.py             SuiteSparse matrix ingestion (auto / manual / githubdata modes)
    train.py              Training (AdamW + cosine annealing, EXPERIMENT-namespaced)
    evaluate.py           Evaluation: Acc, MP, MR, F1, top-k
    ensemble_evaluate.py  Ensemble of independently trained dual-channel models
    render.py             Re-render images from an existing base dataset
    predict.py            Inference script
    benchmark.py          Per-matrix solver benchmarking
    docker-compose.yml

  mm_baseline/            MM-AutoSolver re-implementation (Xiong et al. 2025)
    mm_model.py           Architecture + 17 features + 128×128 density image
    mm_generate.py        Data generation
    mm_train.py           Training (Adam, lr=1e-3, 256 epochs, batch=512)
    mm_evaluate.py        Evaluation: Acc, MP, MR, F1 (paper's four metrics)
    mm_ingest.py          SuiteSparse ingestion
    mm_trim.py            Dataset trimming
    check_distribution.py Per-class sample counts and dataset statistics
    docker-compose.yml

  shared/
    generators.py         Shared synthetic matrix generators
    matrix_io.py          Shared .mtx loading and classification helpers
    cache/                Shared SuiteSparse download cache

container/
  Dockerfile              Shared base image (PETSc + Python dependencies)
  requirements.txt        Python packages installed inside the container

thesis/workdir/           LaTeX thesis source (TUM template)
  chapters/               One .tex file per chapter
  pages/                  Cover, title, disclaimer, abstract, etc.
  bibliography.bib
  main.tex                Root document
  settings.tex            Packages and style config

thesis/BachelorThesis.pdf  Final submitted PDF
requirements.txt           Python packages for local (non-Docker) setup
```

---

## Docker pipeline (recommended)

Docker handles PETSc and all Python dependencies automatically.  Install
[Docker](https://docs.docker.com/get-docker/) and
[Docker Compose](https://docs.docker.com/compose/) first.

### Main experiments (`experiments/thesis_experiments/`)

```bash
cd experiments/thesis_experiments

# Step 1 — generate training data (writes to ./data/)
docker compose run datagen

# Step 1b — ingest SuiteSparse matrices (optional, appends to ./data/dataset.h5)
docker compose run ingest

# Step 2 — train
EXPERIMENT=my_run IMAGE_MODE=magnitude docker compose run trainer

# Step 3 — evaluate
EXPERIMENT=my_run docker compose run evaluate

# Step 3b — ensemble evaluation (four dual-channel models)
docker compose run ensemble_evaluate

# Step 4 — inference on a new matrix
EXPERIMENT=my_run docker compose run predict

# Monitor training live
docker compose up tensorboard   # open http://localhost:6006
```

For running a full grid of experiments (all modes × sizes), use `run_experiments.sh`
instead of calling `docker compose run` manually:

```bash
# All 14 modes × 64px + 128px, 256 epochs (full Campaign 1/2 run)
./run_experiments.sh

# Dual-channel pairs on top of an existing base dataset
SKIP_DATAGEN=1 MODES="" SIZES="" \
DUAL_MODES="magnitude+signed_magnitude magnitude+symmetry" DUAL_SIZES="64" \
  ./run_experiments.sh

# Quick test run
N_SAMPLES=50 MAX_EPOCHS=20 MODES="magnitude" SIZES="64" ./run_experiments.sh
```

See `experiments/thesis_experiments/Documentation.md` for the full variable reference.

Key environment variables for the trainer:

| Variable | Default | Description |
|---|---|---|
| `EXPERIMENT` | `default` | Run name; checkpoints/logs go to `checkpoints/<EXPERIMENT>/` |
| `IMAGE_MODE` | `binary` | Image encoding (see [Image encodings](#image-encodings)) |
| `IMAGE_MODE2` | _(none)_ | Second channel for dual-channel CNN; leave empty for single-channel |
| `IMAGE_SIZE` | `64` | Image resolution in pixels |
| `MODEL_SIZE` | `small` | `small` (~0.5 M params) or `large` (~2.1 M params) |
| `N_SAMPLES` | `10000` | Synthetic training matrices to generate |
| `MAX_EPOCHS` | `100` | Training epochs |
| `BATCH_SIZE` | `256` | Samples per step |
| `LEARNING_RATE` | `3e-4` | Initial AdamW learning rate |
| `CONVERGENCE_PENALTY` | `0.0` | λ for the divergence penalty term |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

GPU support: uncomment the `deploy.resources.reservations` block in `docker-compose.yml`.

### MM-AutoSolver baseline (`experiments/mm_baseline/`)

```bash
cd experiments/mm_baseline

# Step 1 — generate baseline dataset (128×128 density images, 17 features)
docker compose run mm_datagen

# Step 2 — train MM-AutoSolver (Adam lr=1e-3, 256 epochs, batch=512)
docker compose run mm_trainer

# Step 3 — evaluate: Acc, MP, MR, F1
docker compose run mm_evaluate

# Monitor training
docker compose up mm_tensorboard   # open http://localhost:6007
```

---

## Local setup (Fedora only)

> The local setup has only been tested on Fedora 44.  The Docker workflow above is
> recommended for all other systems.

### 1. System dependencies

```bash
sudo dnf install openblas-devel python3-devel
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python packages

petsc4py must be installed **after** PETSc and is excluded from `requirements.txt`.
Install it separately after everything else:

```bash
# Install all other dependencies
pip install -r requirements.txt

# Install PETSc first, then petsc4py against it
pip install petsc==3.25.1
pip install petsc4py==3.25.1 --no-build-isolation
```

### 4. Run scripts directly

Scripts live in `experiments/thesis_experiments/` and `experiments/mm_baseline/`.
Environment variables replace the Docker compose defaults:

```bash
cd experiments/thesis_experiments

# Generate data
N_SAMPLES=5000 IMAGE_MODE=magnitude python generate.py

# Train
EXPERIMENT=local_run IMAGE_MODE=magnitude python train.py

# Evaluate
EXPERIMENT=local_run python evaluate.py
```

---

## Image encodings

14 encoding modes are evaluated, each producing a single-channel image of the sparse matrix:

| Mode | Pixel value |
|---|---|
| `binary` | 1 if any non-zero in block, else 0 |
| `density` | non-zero count per block / block area |
| `log_density` | log(1 + density), normalised |
| `magnitude` | mean absolute entry value per block, normalised |
| `sign` | mean sign per block |
| `signed_magnitude` | signed mean entry per block, normalised |
| `symmetry` | local symmetry score per block |
| `diagonal` | emphasis on diagonal blocks |
| `rcm_binary` | `binary` after Reverse Cuthill–McKee reordering |
| `rcm_density` | `density` after RCM |
| `rcm_log_density` | `log_density` after RCM |
| `rcm_magnitude` | `magnitude` after RCM |
| `rcm_sign` | `sign` after RCM |
| `rcm_signed_magnitude` | `signed_magnitude` after RCM |

Dual-channel experiments set both `IMAGE_MODE` and `IMAGE_MODE2` to use two encodings in parallel.

---

## Experimental campaigns

| Campaign | Name | Features | Model | Image modes | Resolution | Penalty |
|---|---|---|---|---|---|---|
| C1 | Single-Channel Baseline | 14 | Small (~0.5M) | All 14 modes | 64 px + 128 px | none |
| C2 | Extended Features + Penalty | 20 | Large (~2M) | All 14 modes | 64 px only | λ=1.0 |
| C3 | Dual-Channel Approach | 14 | Large (~2M) | 7 modes → 21 pairwise combos | 64 px only | none |
| C4 | Dual-Channel + Extended Features | 20 | Large (~2M) | selected pairs → 30 models | 64 px + 128 px | λ=1.0 |
| C5 | Ensemble | 20 | — | 4 C4 models averaged | 128 px | — |

**C1** establishes a clean baseline: 14 image modes each at 64 px and 128 px, small model, no penalty.

**C2** addresses weaknesses found in C1 by expanding features from 14 to 20 and adding the convergence penalty (λ=1.0) to reduce failure rates. Large model, 64 px only.

**C3** introduces the dual-channel CNN. Six high-performing modes from C1 plus the `sign` mode (7 total) are combined into all 21 pairwise combinations. Large model, 64 px, no penalty.

**C4** combines dual-channel with C2's improvements (20 features, λ=1.0, large model). Mode pairs are selected based on C2 F1 performance and run at both 64 px and 128 px, giving 15 combinations × 2 sizes = 30 models.

**C5** averages the softmax outputs of independently trained models (no retraining). Two ensembles are evaluated:

C3 ensemble (four members, all 64 px):
- `magnitude + signed_magnitude`
- `magnitude + rcm_signed_magnitude`
- `magnitude + symmetry`
- `rcm_magnitude + signed_magnitude`

C4 ensemble (four members, all 128 px):
- `magnitude + log_density`
- `magnitude + rcm_log_density`
- `rcm_magnitude + rcm_signed_magnitude`
- `symmetry + log_density`

---

## Key results

| Method | Acc % | Macro F1 % |
|---|---|---|
| MM-AutoSolver (paper, Xiong et al. 2025) | 78.54 | 62.53 |
| C1 best: `magnitude_64` | 64.19 | 63.42 |
| C2 best: `magnitude_64` | 64.29 | 63.38 |
| C5 ensemble (four C4 models, 128 px) | — | **66.77** |

The C5 ensemble surpasses the MM-AutoSolver approach by approximately 4 percentage points
in macro F1, on a harder and more balanced dataset (9,711 matrices, capped at 600 per class).
The lower accuracy relative to the paper reflects the harder dataset rather than worse prediction quality.

---

## Solver-preconditioner pairs (19 classes)

The 19 classification targets are PETSc KSP+PC combinations that emerged as dataset winners:

`cr+ilu`, `cg+eisenstat`, `cg+bjacobi`, `fbcgsr+jacobi`, `gmres+gamg`, `fgmres+gamg`,
`cg+ilu`, `cr+jacobi`, `minres+gamg`, `fbcgsr+ilu`, `cr+eisenstat`, `bcgsl+none`,
`symmlq+icc`, `bcgsl+asm`, `dgmres+none`, `cgs+gamg`, `fcg+gamg`, `symmlq+jacobi`, `symmlq+sor`

---

## References

Full bibliography: `thesis/workdir/bibliography.bib`.
