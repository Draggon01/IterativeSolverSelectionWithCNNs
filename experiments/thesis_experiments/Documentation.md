# thesis_experiments — Documentation

## Current repository structure

```
model.py              Shared architecture, features, image rendering
generate.py           Synthetic data generation
ingest.py             SuiteSparse ingestion (auto / manual / githubdata)
train.py              Training (AdamW + cosine annealing)
evaluate.py           Evaluation: Acc, MP, MR, F1, top-k, failure rate
ensemble_evaluate.py  Ensemble of dual-channel models
render.py             Image rendering — two modes in one script:
                        single/dual: IMAGE_MODE + optional IMAGE_MODE2
                        multimode:   set MODES="m1 m2 ..." env var
predict.py            Inference on a new matrix

utils/                One-off dataset utilities (merge, trim, drop_rows, ...)
analysis/             Inspection tools (benchmark, visualize, browse, baselines)
tmp/                  Smoke-test scripts (not part of the pipeline)
experimentResults/    Saved results per campaign
```

## Key environment variables

### IMAGE_MODE2 — dual-channel second channel

`IMAGE_MODE2` is a **render-time** parameter. Set it when calling `render`:

```bash
IMAGE_MODE=magnitude IMAGE_MODE2=signed_magnitude docker compose run render
```

`render.py` writes a second `images2` dataset into the output HDF5 alongside
`images`. `train.py` detects dual-channel automatically by checking whether
`images2` exists in the dataset — `IMAGE_MODE2` does not need to be set at
training time.

The checkpoint stores `image_mode2` so `ensemble_evaluate.py` can later
infer which mode to load from the multimode HDF5.

### run_experiments.sh — full experiment loop

`run_experiments.sh` is the main entry point for running experiments. It
handles the full loop of datagen → render → train → evaluate for all
mode/size combinations in one command, and writes all results to a single
summary file.

```bash
# Full grid: all 14 modes × 64px + 128px, 256 epochs
./run_experiments.sh

# Quick subset
MODES="magnitude binary" SIZES="64" MAX_EPOCHS=20 N_SAMPLES=500 ./run_experiments.sh

# Skip datagen if base dataset already exists
SKIP_DATAGEN=1 MODES="magnitude" SIZES="64 128" ./run_experiments.sh

# Dual-channel only (no single-channel), reuse existing data
SKIP_DATAGEN=1 MODES="" SIZES="" \
DUAL_MODES="magnitude+signed_magnitude magnitude+symmetry" DUAL_SIZES="64" \
  ./run_experiments.sh
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODES` | `binary density log_density magnitude` | Single-channel image modes to loop over |
| `SIZES` | `64 128 256 512` | Image resolutions to loop over |
| `DUAL_MODES` | _(none)_ | Dual-channel pairs, e.g. `"magnitude+signed_magnitude magnitude+symmetry"` |
| `DUAL_SIZES` | same as `SIZES` | Resolutions for dual-channel runs |
| `N_SAMPLES` | `10000` | Matrices to generate (ignored if `SKIP_DATAGEN=1`) |
| `MAX_EPOCHS` | `256` | Training epochs per experiment |
| `BATCH_SIZE` | `512` | Training batch size |
| `MODEL_SIZE` | `small` | `small` or `large` |
| `CONVERGENCE_PENALTY` | `0.0` | λ for divergence penalty |
| `SKIP_DATAGEN` | `0` | Skip base data generation |
| `SKIP_RENDER` | `0` | Skip image rendering |
| `SKIP_TRAIN` | `0` | Skip training |
| `SKIP_EVAL` | `0` | Skip evaluation |
| `RESULTS_FILE` | `results_summary.txt` | Output file for all results |

### DUAL_MODES — dual-channel loop in run_experiments.sh

`run_experiments.sh` accepts `DUAL_MODES` as a space-separated list of
`mode1+mode2` pairs (the `+` separates the two channels):

```bash
DUAL_MODES="magnitude+signed_magnitude magnitude+symmetry" ./run_experiments.sh
```

Each pair renders both channels in one pass and trains one model.
This is the only place where the `+` notation is used — `IMAGE_MODE2`
itself is always a plain single-mode string.

### MODES — multimode HDF5 rendering

When `MODES` is set, `render.py` switches to multimode mode and writes
one `images_<mode>` dataset per mode into a shared HDF5. This is used
by `ensemble_evaluate.py`:

```bash
MODES="magnitude signed_magnitude symmetry" docker compose run render_multimode
```

---

## Development log

### Step 1 — Generate initial dataset

```bash
N_SAMPLES=20006 MAX_ITER=5000 STORE_MATRIX=1 SEED=1234 docker compose run -d datagen
```

Solver win distribution (20006 samples):

```
Rank  Solver             Wins      %
────────────────────────────────────
  1   fbcgsr+ilu         3131   15.7%
  2   fbcgsr+jacobi      2945   14.7%
  3   cg+eisenstat       2692   13.5%
  4   cr+jacobi          1833    9.2%
  5   cg+ilu             1827    9.1%
  6   cg+bjacobi         1323    6.6%
  7   minres+gamg        1268    6.3%
  8   cr+ilu              975    4.9%
  9   gmres+gamg          824    4.1%
 10   cr+eisenstat        817    4.1%
 11   bcgsl+none          801    4.0%
 12   symmlq+icc          460    2.3%
 13   fgmres+gamg         343    1.7%
 14   cgs+gamg            199    1.0%
 15   bcgsl+asm           182    0.9%
 16   fcg+gamg            168    0.8%
 17   symmlq+sor           90    0.4%
 18   symmlq+jacobi        79    0.4%
 19   dgmres+none          49    0.2%

Matrix sizes — min=99  median=7420  mean=13156  max=89999
```

### Step 2 — Initial test run

```bash
./run_experiments.sh
```

### Step 3 — Ingest SuiteSparse githubdata matrices

```bash
CACHE_DIR=../shared/cache/ \
DATA_DIR=./data/saves/ex1/suite_githubdata/ \
SOLVER_TIMEOUT=80 \
MAX_ITER=250000 \
  docker compose run -d githubdata_ingest
```

### Step 4 — Add further SuiteSparse matrices

### Step 5 — Run experiments for each campaign