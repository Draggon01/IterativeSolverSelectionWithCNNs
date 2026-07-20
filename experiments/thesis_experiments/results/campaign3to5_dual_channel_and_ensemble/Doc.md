# combinedwithWinnersDataFolder

Results from dual-channel (2-stream CNN) experiments using the large model.

Each experiment combines two image rendering modes as separate CNN input channels.
Architecture: `MODEL_SIZE=large` (3 ResBlocks, 128 channels, ~2M params).
Dataset: same base dataset as combinedDataFolder (SKIP_DATAGEN=1).
Image size: 64px. Experiment dirs: `data/<mode1>__<mode2>_64/`.

## Run command

21 pairwise combinations of the 7 best-performing single-channel modes
(best 6 from the grid search + sign added for curiosity):

```bash
SKIP_DATAGEN=1 \
CACHE_DIR=../shared/cache/ \
RESULTS_FILE=./experimentResults/combinedwithWinnersDataFolder/cwWDFsummary.txt \
MODES="" SIZES="" \
DUAL_MODES="magnitude+rcm_signed_magnitude \
magnitude+density \
magnitude+rcm_magnitude \
magnitude+symmetry \
magnitude+signed_magnitude \
magnitude+sign \
rcm_signed_magnitude+density \
rcm_signed_magnitude+rcm_magnitude \
rcm_signed_magnitude+symmetry \
rcm_signed_magnitude+signed_magnitude \
rcm_signed_magnitude+sign \
density+rcm_magnitude \
density+symmetry \
density+signed_magnitude \
density+sign \
rcm_magnitude+symmetry \
rcm_magnitude+signed_magnitude \
rcm_magnitude+sign \
symmetry+signed_magnitude \
symmetry+sign \
signed_magnitude+sign" \
DUAL_SIZES="64" \
MODEL_SIZE=large \
BATCH_SIZE=64 \
./run_experiments.sh
```

## Dual-channel results (large model, 64px)

Sorted by Acc% descending.

| Experiment                              | Acc%  | MP%   | MR%   | F1%   |
|-----------------------------------------|-------|-------|-------|-------|
| magnitude__signed_magnitude_64          | 66.05 | 68.21 | 68.23 | 65.32 |
| magnitude__rcm_signed_magnitude_64      | 66.05 | 69.25 | 67.64 | 64.79 |
| magnitude__symmetry_64                  | 65.63 | 67.20 | 67.59 | 64.66 |
| rcm_magnitude__signed_magnitude_64      | 65.63 | 68.90 | 67.61 | 64.86 |
| rcm_magnitude__sign_64                  | 64.91 | 66.97 | 67.04 | 64.03 |
| magnitude__density_64                   | 64.50 | 68.09 | 67.13 | 63.95 |
| magnitude__sign_64                      | 64.50 | 66.79 | 66.93 | 63.52 |
| magnitude__rcm_magnitude_64             | 64.40 | 66.64 | 65.89 | 63.24 |
| density__rcm_magnitude_64               | 64.40 | 67.28 | 65.93 | 63.72 |
| density__sign_64                        | 63.98 | 66.90 | 66.42 | 63.14 |
| rcm_signed_magnitude__rcm_magnitude_64  | 63.88 | 66.95 | 65.97 | 63.22 |
| density__symmetry_64                    | 63.57 | 63.78 | 66.09 | 62.98 |
| rcm_signed_magnitude__density_64        | 63.36 | 65.94 | 65.10 | 61.89 |
| symmetry__signed_magnitude_64           | 63.36 | 64.72 | 64.92 | 62.08 |
| rcm_magnitude__symmetry_64              | 64.29 | 66.85 | 66.74 | 63.34 |
| symmetry__sign_64                       | 63.05 | 65.83 | 65.22 | 61.92 |
| rcm_signed_magnitude__symmetry_64       | 62.85 | 63.05 | 64.26 | 61.81 |
| rcm_signed_magnitude__sign_64           | 62.44 | 65.12 | 64.14 | 60.79 |
| signed_magnitude__sign_64               | 62.33 | 64.70 | 63.94 | 61.29 |
| rcm_signed_magnitude__signed_magnitude_64 | 62.33 | 64.34 | 63.62 | 60.55 |
| density__signed_magnitude_64            | 61.82 | 64.35 | 63.66 | 60.52 |

**Top 4 by F1:** magnitude__signed_magnitude (65.32), rcm_magnitude__signed_magnitude (64.86), magnitude__rcm_signed_magnitude (64.79), magnitude__symmetry (64.66)

## Ensemble evaluation results

<!-- Fill in after running ensemble_evaluate -->

| Members                                                                                                                | Acc%  | MP%   | MR%   | F1%   |
|------------------------------------------------------------------------------------------------------------------------|-------|-------|-------|-------|
| magnitude__signed_magnitude + magnitude__rcm_signed_magnitude + magnitude__symmetry + rcm_magnitude__signed_magnitude  | 67.39 | 69.47 | 69.58 | 66.37 |

Gain over best individual model: +1.34pp Acc, +1.05pp F1.

Notable per-class results from ensemble:

| Solver          | F1%   | Note                                      |
|-----------------|-------|-------------------------------------------|
| cr+eisenstat    | 85.5  | best individual class                     |
| minres+gamg     | 88.7  | highest recall (98.3%)                    |
| gmres+gamg      | 20.6  | still very hard — confused with fgmres    |
| fcg+gamg        | 52.3  | low precision (39%), over-predicted       |
| fbcgsr+ilu      | 58.9  | low recall (46.7%)                        |

## Mode ranking (single-channel reference, small model, 64px)

| Mode                  | Acc%  | F1%   |
|-----------------------|-------|-------|
| magnitude             | 64.19 | 63.42 |
| rcm_signed_magnitude  | 63.67 | 62.24 |
| density (128px)       | 63.26 | 62.25 |
| rcm_magnitude         | 63.05 | 62.55 |
| symmetry              | 62.95 | 61.85 |
| signed_magnitude      | 62.85 | 61.68 |
| sign                  | 60.37 | 59.29 |
| 5-NN baseline         | 62.95 | 61.12 |
| nocnn baseline        | 58.31 | 57.57 |

## Ensemble run command

Rendering and evaluation are fully automatic — just specify the experiments and run:

```bash
EXPERIMENTS="magnitude__signed_magnitude_64 magnitude__rcm_signed_magnitude_64 magnitude__symmetry_64 rcm_magnitude__signed_magnitude_64" \
docker compose run --rm ensemble_evaluate
```

The script automatically determines which image modes are needed from the checkpoints,
renders only the missing ones into `data/multimode/dataset.h5`, then runs evaluation.
Re-running with the same experiments skips rendering entirely.
Adding new experiments only renders the additionally required modes.

Fill in the `?` values in the ensemble results table above after running this.
