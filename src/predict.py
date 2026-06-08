"""
predict.py — Recommend a Krylov solver for a new sparse matrix system.

Loads the latest checkpoint and outputs the predicted best solver together
with per-class probabilities.

Environment variables:
  CHECKPOINT_DIR   Directory containing epoch_*.pt files  (default /workspace/checkpoints)
  MATRIX_PATH      Path to a scipy .npz sparse matrix file (optional).
                   If unset, a random example matrix is generated instead.
  DEVICE           "cpu", "cuda", or "auto"               (default auto)
"""

import os
import logging

import numpy as np
import scipy.sparse as sp
import torch

from model import (
    matrix_features, sparsity_image,
    SOLVER_NAMES, load_checkpoint,
    SolverSelectorNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MATRIX_PATH = os.getenv("MATRIX_PATH", "")

_dev   = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)


# ── inference ─────────────────────────────────────────────────────────────────

def predict_solver(
    model: SolverSelectorNet,
    A: sp.csr_matrix,
    image_mode: str = "binary",
) -> tuple[str, np.ndarray]:
    """
    Return (recommended_solver_name, probability_array).

    probability_array[i] corresponds to SOLVER_NAMES[i].
    """
    feat = torch.from_numpy(matrix_features(A)).unsqueeze(0).to(DEVICE)
    img  = torch.from_numpy(sparsity_image(A, mode=image_mode)).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img, feat)

    probs     = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    best_idx  = int(probs.argmax())
    return SOLVER_NAMES[best_idx], probs


# ── example matrix (fallback when MATRIX_PATH is not set) ─────────────────────

def _example_matrix() -> tuple[sp.csr_matrix, str]:
    """Generate a small random example matrix for demonstration."""
    # Import lazily to keep this module PETSc-free
    from generate_data import sample_matrix
    rng = np.random.default_rng(0)
    A, mat_type = sample_matrix(rng)
    return A, mat_type


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    model, ckpt = load_checkpoint(DEVICE)
    log.info(
        "Loaded checkpoint: epoch=%d  val_acc=%.4f",
        ckpt.get("epoch", -1), ckpt.get("val_acc", float("nan")),
    )

    if MATRIX_PATH and os.path.exists(MATRIX_PATH):
        A = sp.load_npz(MATRIX_PATH).tocsr().astype(np.float64)
        log.info("Loaded matrix from %s  shape=%s  nnz=%d", MATRIX_PATH, A.shape, A.nnz)
    else:
        A, mat_type = _example_matrix()
        log.info(
            "No MATRIX_PATH set — using a random example matrix  "
            "type=%s  shape=%s  nnz=%d",
            mat_type, A.shape, A.nnz,
        )

    image_mode = ckpt.get("image_mode", "binary")
    solver, probs = predict_solver(model, A, image_mode)

    print(f"\nImage mode: {image_mode}")
    print("Solver probabilities (top 10):")
    for name, p in sorted(zip(SOLVER_NAMES, probs), key=lambda x: -x[1])[:10]:
        bar = "█" * int(p * 30)
        print(f"  {name:<14s}  {p:.4f}  {bar}")

    print(f"\nRecommended: {solver}\n")


if __name__ == "__main__":
    main()
