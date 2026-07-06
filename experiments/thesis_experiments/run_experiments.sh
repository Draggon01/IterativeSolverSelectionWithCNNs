#!/usr/bin/env bash
# run_experiments.sh — Run all IMAGE_MODE × IMAGE_SIZE combinations sequentially.
#
# Workflow:
#   1. Generate data ONCE into ./data/base/ with STORE_MATRIX=1
#      (runs PETSc solvers — the slow step, done only once)
#   1b. Train + evaluate features-only baseline (no CNN)
#   2. For each mode × size combination:
#      a. Render images from the base dataset (fast, no solver re-runs)
#      b. Train  — checkpoints/logs go to ./checkpoints/<mode>_<size>/
#      c. Evaluate — results appended to RESULTS_FILE
#
# Usage:
#   ./run_experiments.sh                      # all 16 combinations + nocnn
#   SKIP_DATAGEN=1 ./run_experiments.sh       # base data already generated
#   MODES="binary density" SIZES="64 128" ./run_experiments.sh   # subset
#   MAX_EPOCHS=5 N_SAMPLES=10 ./run_experiments.sh               # quick test
#   MODES="" SIZES="" ./run_experiments.sh    # nocnn baseline only
#
# Environment variables:
#   MODES          Space-separated image modes    (default: binary density log_density magnitude)
#                  Extra modes: symmetry diagonal sign signed_magnitude
#                               rcm_binary rcm_density rcm_log_density rcm_magnitude
#                               rcm_sign rcm_signed_magnitude
#   SIZES          Space-separated image sizes    (default: 64 128 256 512)
#   N_SAMPLES      Samples for base datagen       (default: 10000)
#   MAX_EPOCHS     Training epochs per experiment (default: 256, matches MM paper)
#   BATCH_SIZE     Training batch size            (default: 512, matches MM paper)
#   LEARNING_RATE  Optimizer learning rate        (default: 0.0003)
#   SKIP_DATAGEN   1 = skip base data generation  (default: 0)
#   SKIP_RENDER    1 = skip image rendering        (default: 0)
#   SKIP_TRAIN     1 = skip training              (default: 0)
#   SKIP_EVAL      1 = skip evaluation            (default: 0)
#   KEEP_DATASET   1 = keep rendered dataset.h5   (default: 0, deleted after eval)
#   DEVICE         cpu | cuda | auto              (default: auto)
#   RESULTS_FILE   output file for all results    (default: results_summary.txt)
#   CACHE_DIR      SuiteSparse .mtx cache for render (default: ../shared/cache)
#   BASE_DATA_DIR  Path to base dataset.h5            (default: ./data/base)

set -euo pipefail
cd "$(dirname "$0")"

MODES="${MODES-binary density log_density magnitude}"
SIZES="${SIZES-64 128 256 512}"

N_SAMPLES="${N_SAMPLES:-10000}"
MIN_PER_CLASS="${MIN_PER_CLASS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-256}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
KEEP_DATASET="${KEEP_DATASET:-0}"
DEVICE="${DEVICE:-auto}"
RESULTS_FILE="${RESULTS_FILE:-./results_summary.txt}"
CACHE_DIR="${CACHE_DIR:-../shared/cache}"
MODEL_SIZE="${MODEL_SIZE:-small}"
CONVERGENCE_PENALTY="${CONVERGENCE_PENALTY:-0.0}"
# DUAL_MODES: space-separated mode pairs for 2-channel experiments, e.g.:
#   DUAL_MODES="magnitude+sign magnitude+signed_magnitude"
# Each pair uses IMAGE_MODE (ch1) + IMAGE_MODE2 (ch2), data goes to data/<m1>__<m2>_<size>/
DUAL_MODES="${DUAL_MODES:-}"
DUAL_SIZES="${DUAL_SIZES:-$SIZES}"

COMPOSE="docker compose"
docker compose down --remove-orphans 2>/dev/null || true
BASE_DATA_DIR="${BASE_DATA_DIR:-$(pwd)/data/base}"

echo "========================================================"
echo " Thesis Experiments — Image Variant Grid Search"
echo "  Modes      : $MODES"
echo "  Sizes      : $SIZES"
echo "  Model size : $MODEL_SIZE"
echo "  Epochs     : $MAX_EPOCHS  |  Datagen N: $N_SAMPLES (ignored when SKIP_DATAGEN=1)"
echo "  Dual modes : ${DUAL_MODES:-(none)}"
echo "  Output     : $RESULTS_FILE"
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
        MIN_PER_CLASS="$MIN_PER_CLASS" \
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

# ── Dataset distribution ─────────────────────────────────────────────────────
echo "[viz] Writing solver win distribution to $RESULTS_FILE ..."
{
    echo "============================================================"
    echo "  Dataset : ${BASE_DATA_DIR}/dataset.h5"
    echo "  Date    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    DATA_DIR="$BASE_DATA_DIR" \
    OUT_DIR="$(pwd)/viz" \
    $COMPOSE run --rm viz 2>/dev/null || echo "  (viz failed — skipping)"
    echo ""
} >> "$RESULTS_FILE"

# ── Non-learning baselines ───────────────────────────────────────────────────
echo "[baselines] Running majority-class and KNN baselines ..."
{
    echo "============================================================"
    echo "  Baselines : majority class, 1-NN, 5-NN"
    echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    DATA_DIR="$BASE_DATA_DIR" \
    $COMPOSE run --rm baselines 2>/dev/null || echo "  (baselines failed — skipping)"
    echo ""
} >> "$RESULTS_FILE"

# Helper: append one experiment's full section to RESULTS_FILE
# Usage: append_results <exp_name> <eval_out>
append_results() {
    local exp_name="$1" eval_out="$2"
    local acc mp mr f1
    acc=$(echo "$eval_out" | grep "Accuracy (Acc)" | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
    mp=$(echo  "$eval_out" | grep "Macro Precision" | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
    mr=$(echo  "$eval_out" | grep "Macro Recall"    | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")
    f1=$(echo  "$eval_out" | grep "Macro F1"        | grep -oP '\d+\.\d+(?=%)' | head -1 || echo "?")

    {
        echo ""
        echo "experiment           Acc%    MP%     MR%     F1%"
        echo "----------------------------------------------------"
        printf "%-20s %7s %7s %7s %7s\n" "$exp_name" "$acc" "$mp" "$mr" "$f1"
        echo ""
        echo "$eval_out"
    } >> "$RESULTS_FILE"
}

# ── Step 1b: Features-only baseline (no CNN) ─────────────────────────────────
echo ""
echo "[nocnn] ────────────────────────────────────────────────────────"
echo "  Features-only baseline — no CNN branch, uses base dataset"
echo "────────────────────────────────────────────────────────────────"

cat >> "$RESULTS_FILE" <<NOCNN_HDR
============================================================
  Experiment : nocnn
  Date       : $(date '+%Y-%m-%d %H:%M:%S')
  NO_CNN     : True  (features-only, no CNN branch)
============================================================
NOCNN_HDR
EXPERIMENT=nocnn IMAGE_MODE=binary IMAGE_SIZE=64 NO_CNN=1 \
    $COMPOSE run --rm meta | grep -A999 "Solver pairs" >> "$RESULTS_FILE"

if [[ "$SKIP_TRAIN" != "1" ]]; then
    echo "  [trainer] Training nocnn ..."
    DATA_DIR="$BASE_DATA_DIR" \
    EXPERIMENT="nocnn" \
    NO_CNN=1 \
    MAX_EPOCHS="$MAX_EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    LEARNING_RATE="$LEARNING_RATE" \
    DEVICE="$DEVICE" \
    $COMPOSE run --rm trainer
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
    echo "  [evaluate] Evaluating nocnn ..."
    eval_out=$(DATA_DIR="$BASE_DATA_DIR" \
               EXPERIMENT="nocnn" \
               DEVICE="$DEVICE" \
               $COMPOSE run --rm evaluate 2>&1) || true
    echo "$eval_out"
    append_results "nocnn" "$eval_out"
fi

echo "  [done] nocnn"

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

        # Write per-experiment metadata header with correct mode/size
        EXPERIMENT="$exp" IMAGE_MODE="$mode" IMAGE_SIZE="$size" \
            $COMPOSE run --rm meta >> "$RESULTS_FILE"

        # ── Render ───────────────────────────────────────────────
        if [[ "$SKIP_RENDER" != "1" ]]; then
            if [[ -f "${data_dir}/dataset.h5" ]]; then
                echo "  [render] dataset.h5 already exists — skipping"
            else
                echo "  [render] Rendering mode=$mode size=$size ..."
                SRC_DATA_DIR="$BASE_DATA_DIR" \
                DATA_DIR="$data_dir" \
                CACHE_DIR="$CACHE_DIR" \
                IMAGE_MODE="$mode" \
                IMAGE_SIZE="$size" \
                $COMPOSE run --rm render
            fi
        fi

        # ── Train ────────────────────────────────────────────────
        if [[ "$SKIP_TRAIN" != "1" ]]; then
            # Scale batch size down for large images to avoid OOM
            if   [[ "$size" -ge 512 ]]; then eff_batch=$(( BATCH_SIZE > 64  ? 64  : BATCH_SIZE ))
            elif [[ "$size" -ge 256 ]]; then eff_batch=$(( BATCH_SIZE > 128 ? 128 : BATCH_SIZE ))
            else                              eff_batch="$BATCH_SIZE"
            fi
            echo "  [trainer] Training EXPERIMENT=$exp  (batch=$eff_batch) ..."
            DATA_DIR="$data_dir" \
            IMAGE_MODE="$mode" \
            IMAGE_SIZE="$size" \
            EXPERIMENT="$exp" \
            MAX_EPOCHS="$MAX_EPOCHS" \
            BATCH_SIZE="$eff_batch" \
            LEARNING_RATE="$LEARNING_RATE" \
            MODEL_SIZE="$MODEL_SIZE" \
            CONVERGENCE_PENALTY="$CONVERGENCE_PENALTY" \
            DEVICE="$DEVICE" \
            $COMPOSE run --rm trainer
        fi

        # ── Evaluate ─────────────────────────────────────────────
        if [[ "$SKIP_EVAL" != "1" ]]; then
            echo "  [evaluate] Evaluating EXPERIMENT=$exp ..."
            eval_out=$(DATA_DIR="$data_dir" \
                       EXPERIMENT="$exp" \
                       DEVICE="$DEVICE" \
                       $COMPOSE run --rm evaluate 2>&1) || true
            echo "$eval_out"
            append_results "$exp" "$eval_out"
        fi

        # ── Cleanup rendered dataset ──────────────────────────────
        if [[ "$KEEP_DATASET" != "1" && -f "${data_dir}/dataset.h5" ]]; then
            echo "  [cleanup] Removing rendered dataset to free disk space ..."
            rm -f "${data_dir}/dataset.h5"
        fi

        echo "  [done] $exp"
    done
done

# ── Step 3: Dual-channel experiments (2-stream CNN) ──────────────────────────
if [[ -n "$DUAL_MODES" ]]; then
    dual_total=0
    for pair in $DUAL_MODES; do for size in $DUAL_SIZES; do dual_total=$((dual_total + 1)); done; done

    dual_current=0
    for pair in $DUAL_MODES; do
        for size in $DUAL_SIZES; do
            dual_current=$((dual_current + 1))
            mode1="${pair%+*}"
            mode2="${pair#*+}"
            exp="${mode1}__${mode2}_${size}"
            data_dir="$(pwd)/data/${exp}"

            echo ""
            echo "[$dual_current/$dual_total] ── Dual-channel ──────────────────────────"
            echo "  Experiment : $exp"
            echo "  Mode 1     : $mode1   Mode 2: $mode2   Size: $size"
            echo "  Model size : $MODEL_SIZE"
            echo "────────────────────────────────────────────────────────────────"

            mkdir -p "$data_dir"

            EXPERIMENT="$exp" IMAGE_MODE="$mode1" IMAGE_SIZE="$size" \
                $COMPOSE run --rm meta >> "$RESULTS_FILE"

            # ── Render (both channels in one pass) ───────────────────
            if [[ "$SKIP_RENDER" != "1" ]]; then
                if [[ -f "${data_dir}/dataset.h5" ]]; then
                    echo "  [render] dataset.h5 already exists — skipping"
                else
                    echo "  [render] Rendering mode1=$mode1 mode2=$mode2 size=$size ..."
                    SRC_DATA_DIR="$BASE_DATA_DIR" \
                    DATA_DIR="$data_dir" \
                    CACHE_DIR="$CACHE_DIR" \
                    IMAGE_MODE="$mode1" \
                    IMAGE_MODE2="$mode2" \
                    IMAGE_SIZE="$size" \
                    $COMPOSE run --rm render
                fi
            fi

            # ── Train ────────────────────────────────────────────────
            if [[ "$SKIP_TRAIN" != "1" ]]; then
                if   [[ "$size" -ge 512 ]]; then eff_batch=$(( BATCH_SIZE > 64  ? 64  : BATCH_SIZE ))
                elif [[ "$size" -ge 256 ]]; then eff_batch=$(( BATCH_SIZE > 128 ? 128 : BATCH_SIZE ))
                else                              eff_batch="$BATCH_SIZE"
                fi
                echo "  [trainer] Training $exp  (batch=$eff_batch, model=$MODEL_SIZE) ..."
                DATA_DIR="$data_dir" \
                IMAGE_MODE="$mode1" \
                IMAGE_MODE2="$mode2" \
                IMAGE_SIZE="$size" \
                EXPERIMENT="$exp" \
                MAX_EPOCHS="$MAX_EPOCHS" \
                BATCH_SIZE="$eff_batch" \
                LEARNING_RATE="$LEARNING_RATE" \
                MODEL_SIZE="$MODEL_SIZE" \
                CONVERGENCE_PENALTY="$CONVERGENCE_PENALTY" \
                DEVICE="$DEVICE" \
                $COMPOSE run --rm trainer
            fi

            # ── Evaluate ─────────────────────────────────────────────
            if [[ "$SKIP_EVAL" != "1" ]]; then
                echo "  [evaluate] Evaluating $exp ..."
                eval_out=$(DATA_DIR="$data_dir" \
                           EXPERIMENT="$exp" \
                           DEVICE="$DEVICE" \
                           $COMPOSE run --rm evaluate 2>&1) || true
                echo "$eval_out"
                append_results "$exp" "$eval_out"
            fi

            # ── Cleanup ──────────────────────────────────────────────
            if [[ "$KEEP_DATASET" != "1" && -f "${data_dir}/dataset.h5" ]]; then
                echo "  [cleanup] Removing rendered dataset ..."
                rm -f "${data_dir}/dataset.h5"
            fi

            echo "  [done] $exp"
        done
    done
fi

echo ""
echo "========================================================"
echo " All experiments complete."
echo " Results → $RESULTS_FILE"
echo "========================================================"
