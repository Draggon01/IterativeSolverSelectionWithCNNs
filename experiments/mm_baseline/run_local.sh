#!/usr/bin/env bash
# run_local.sh — Run the MM-AutoSolver baseline without Docker.
#
# Prerequisites (install once):
#   pip install "petsc>=3.20"
#   pip install --no-build-isolation "petsc4py>=3.20"
#   pip install torch scipy h5py tensorboard
#
# Note: petsc4py is only required for the data generation step.
#       Training and evaluation are pure PyTorch and work with any Python env.
#
# Usage:
#   cd baseline/
#   bash run_local.sh          # full pipeline (generate + ingest + train + evaluate)
#   bash run_local.sh generate # synthetic data generation only
#   bash run_local.sh ingest   # SuiteSparse ingestion only (appends to existing dataset.h5)
#   bash run_local.sh train    # training only (needs ./data/dataset.h5)
#   bash run_local.sh evaluate # evaluation only (needs ./checkpoints/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output directories (created automatically by the scripts)
export DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRIPT_DIR/checkpoints}"
export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

# Hyperparameters — override via environment before calling this script
export N_SAMPLES="${N_SAMPLES:-5000}"
export N_MATRICES="${N_MATRICES:-1000}"
export MODE="${MODE:-auto}"
export MAX_EPOCHS="${MAX_EPOCHS:-256}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export LEARNING_RATE="${LEARNING_RATE:-0.001}"
export VAL_SPLIT="${VAL_SPLIT:-0.10}"
export DEVICE="${DEVICE:-auto}"

STEP="${1:-all}"

run_generate() {
    echo "==> Step 1: Generating dataset (N_SAMPLES=$N_SAMPLES) ..."
    python mm_generate.py
}

run_ingest() {
    echo "==> Step 1b: Ingesting SuiteSparse matrices (N_MATRICES=$N_MATRICES, MODE=$MODE) ..."
    python mm_ingest.py
}

run_train() {
    echo "==> Step 2: Training MM-AutoSolver (epochs=$MAX_EPOCHS, batch=$BATCH_SIZE, lr=$LEARNING_RATE) ..."
    python mm_train.py
}

run_evaluate() {
    echo "==> Step 3: Evaluating (Acc / Macro-P / Macro-R / Macro-F1) ..."
    python mm_evaluate.py
}

case "$STEP" in
    generate) run_generate ;;
    ingest)   run_ingest   ;;
    train)    run_train    ;;
    evaluate) run_evaluate ;;
    all)
        run_generate
        run_ingest
        run_train
        run_evaluate
        ;;
    *)
        echo "Unknown step '$STEP'. Use: all | generate | ingest | train | evaluate"
        exit 1
        ;;
esac
