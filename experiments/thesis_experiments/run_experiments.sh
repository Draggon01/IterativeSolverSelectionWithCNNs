#!/usr/bin/env bash
# run_experiments.sh — Run all IMAGE_MODE × IMAGE_SIZE combinations sequentially.
#
# Workflow:
#   1. Generate data ONCE into ./data/base/ with STORE_MATRIX=1
#      (runs PETSc solvers — the slow step, done only once)
#   2. For each mode × size combination:
#      a. Render images from the base dataset (fast, no solver re-runs)
#      b. Train  — checkpoints/logs go to ./checkpoints/<mode>_<size>/
#      c. Evaluate — Acc, MP, MR, F1 written to results_summary.txt
#
# Usage:
#   ./run_experiments.sh                      # all 16 combinations
#   SKIP_DATAGEN=1 ./run_experiments.sh       # base data already generated
#   MODES="binary density" SIZES="64 128" ./run_experiments.sh   # subset
#   MAX_EPOCHS=5 N_SAMPLES=10 ./run_experiments.sh               # quick test
#
# Environment variables:
#   MODES          Space-separated image modes    (default: binary density log_density magnitude)
#   SIZES          Space-separated image sizes    (default: 64 128 256 512)
#   N_SAMPLES      Samples for base datagen       (default: 10000)
#   MAX_EPOCHS     Training epochs per experiment (default: 100)
#   BATCH_SIZE     Training batch size            (default: 256)
#   LEARNING_RATE  Optimizer learning rate        (default: 0.0003)
#   SKIP_DATAGEN   1 = skip base data generation  (default: 0)
#   SKIP_RENDER    1 = skip image rendering        (default: 0)
#   SKIP_TRAIN     1 = skip training              (default: 0)
#   SKIP_EVAL      1 = skip evaluation            (default: 0)
#   DEVICE         cpu | cuda | auto              (default: auto)

set -euo pipefail
cd "$(dirname "$0")"

MODES="${MODES:-binary density log_density magnitude}"
SIZES="${SIZES:-64 128 256 512}"

N_SAMPLES="${N_SAMPLES:-10000}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DEVICE="${DEVICE:-auto}"

COMPOSE="docker compose"
BASE_DATA_DIR="$(pwd)/data/base"
RESULTS_FILE="./results_summary.txt"

echo "========================================================"
echo " Thesis Experiments — Image Variant Grid Search"
echo "  Modes : $MODES"
echo "  Sizes : $SIZES"
echo "  Epochs: $MAX_EPOCHS  |  Samples: $N_SAMPLES"
echo "========================================================"
echo ""

# ── Step 1: Generate base dataset once ───────────────────────────────────────
if [[ "$SKIP_DATAGEN" != "1" ]]; then
    if [[ -f "${BASE_DATA_DIR}/dataset.h5" ]]; then
        echo "[datagen] ${BASE_DATA_DIR}/dataset.h5 already exists — skipping"
        echo "          (set SKIP_DATAGEN=1 to suppress this check, or delete the file to regenerate)"
    else
        echo "[datagen] Generating $N_SAMPLES samples into data/base/ ..."
        mkdir -p "$BASE_DATA_DIR"
        DATA_DIR="$BASE_DATA_DIR" \
        N_SAMPLES="$N_SAMPLES" \
        STORE_MATRIX=1 \
        $COMPOSE run --rm datagen
        echo "[datagen] Done."
    fi
else
    echo "[datagen] SKIP_DATAGEN=1 — skipping"
fi

# Validate base dataset exists before continuing
if [[ ! -f "${BASE_DATA_DIR}/dataset.h5" ]]; then
    echo "ERROR: Base dataset not found at ${BASE_DATA_DIR}/dataset.h5"
    echo "       Run without SKIP_DATAGEN=1 first, or check your DATA_DIR."
    exit 1
fi

# Write results header
cat > "$RESULTS_FILE" <<'HDR'
experiment           Acc%    MP%     MR%     F1%
----------------------------------------------------
HDR

# ── Step 2: Render + Train + Evaluate each combination ───────────────────────
total=0
for mode in $MODES; do for size in $SIZES; do total=$((total + 1)); done; done

current=0
for mode in $MODES; do
    for size in $SIZES; do
        current=$((current + 1))
        exp="${mode}_${size}"
        data_dir="$(pwd)/data/${exp}"

        echo ""
        echo "[$current/$total] ────────────────────────────────────────────"
        echo "  Experiment : $exp"
        echo "  Data dir   : $data_dir"
        echo "────────────────────────────────────────────────────────────────"

        mkdir -p "$data_dir"

        # ── Render ───────────────────────────────────────────────
        if [[ "$SKIP_RENDER" != "1" ]]; then
            if [[ -f "${data_dir}/dataset.h5" ]]; then
                echo "  [render] dataset.h5 already exists — skipping"
            else
                echo "  [render] Rendering mode=$mode size=$size ..."
                SRC_DATA_DIR="$BASE_DATA_DIR" \
                DATA_DIR="$data_dir" \
                IMAGE_MODE="$mode" \
                IMAGE_SIZE="$size" \
                $COMPOSE run --rm render
            fi
        fi

        # ── Train ────────────────────────────────────────────────
        if [[ "$SKIP_TRAIN" != "1" ]]; then
            echo "  [trainer] Training EXPERIMENT=$exp ..."
            DATA_DIR="$data_dir" \
            IMAGE_MODE="$mode" \
            IMAGE_SIZE="$size" \
            EXPERIMENT="$exp" \
            MAX_EPOCHS="$MAX_EPOCHS" \
            BATCH_SIZE="$BATCH_SIZE" \
            LEARNING_RATE="$LEARNING_RATE" \
            DEVICE="$DEVICE" \
            $COMPOSE run --rm trainer
        fi

        # ── Evaluate ─────────────────────────────────────────────
        if [[ "$SKIP_EVAL" != "1" ]]; then
            echo "  [evaluate] Evaluating EXPERIMENT=$exp ..."
            eval_out=$(DATA_DIR="$data_dir" \
                       EXPERIMENT="$exp" \
                       DEVICE="$DEVICE" \
                       $COMPOSE run --rm evaluate 2>&1)
            echo "$eval_out"

            acc=$(echo "$eval_out" | grep "Accuracy (Acc)" | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
            mp=$(echo  "$eval_out" | grep "Macro Precision" | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
            mr=$(echo  "$eval_out" | grep "Macro Recall"    | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
            f1=$(echo  "$eval_out" | grep "Macro F1"        | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
            printf "%-20s %7s %7s %7s %7s\n" "$exp" "$acc" "$mp" "$mr" "$f1" >> "$RESULTS_FILE"
        fi

        echo "  [done] $exp"
    done
done

echo ""
echo "========================================================"
echo " All $total experiments complete."
echo " Results summary → $RESULTS_FILE"
echo "========================================================"
echo ""
cat "$RESULTS_FILE"
