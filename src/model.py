"""
model.py — Shared utilities for solver selection.

Imported by generate_data, train_solver_selector, predict, and
benchmark_solvers so that no script needs to import another.
"""

import os
import glob
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# ── constants ─────────────────────────────────────────────────────────────────

# PETSc KSP type strings used as class labels throughout the project.
# "bcgs" is PETSc's internal name for BiCGSTAB.
SOLVERS    = ["cg", "gmres", "bicg", "bcgs", "tfqmr", "minres"]
N_SOLVERS  = len(SOLVERS)
SOLVER_IDX = {s: i for i, s in enumerate(SOLVERS)}

N_FEATURES = 8
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "64"))

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints")


# ── feature extraction ────────────────────────────────────────────────────────

def matrix_features(A: sp.csr_matrix) -> np.ndarray:
    """
    Return an (N_FEATURES,) float32 array of scalar matrix statistics.

    Features:
        0  log(1 + n)                 — log-scaled matrix size
        1  log(1 + nnz)               — log-scaled non-zero count
        2  nnz / n²                   — fill ratio (density)
        3  ‖A − Aᵀ‖_F / ‖A‖_F         — 0 = symmetric, ~1 = fully asymmetric
        4  mean(|diag| / Σ|row|)      — diagonal dominance ratio (clipped)
        5  ‖A‖_F / n                  — size-normalised Frobenius norm
        6  tr(A) / n                  — size-normalised trace
        7  max|a_ij| / mean|a_ij|     — relative maximum entry magnitude
    """
    n   = A.shape[0]
    nnz = A.nnz

    frob     = sp.linalg.norm(A, "fro")
    sym_norm = sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12)

    diag    = np.abs(A.diagonal())
    offsum  = np.array(np.abs(A).sum(axis=1)).ravel() - diag
    dom     = float(np.mean(diag / (offsum + 1e-12)))

    vals    = A.data if nnz > 0 else np.array([0.0])
    max_abs  = float(np.abs(vals).max())
    mean_abs = float(np.abs(vals).mean())

    return np.array([
        np.log1p(n),
        np.log1p(nnz),
        nnz / (n * n),
        float(sym_norm),
        float(np.clip(dom, 0.0, 20.0)),
        frob / (n + 1e-12),
        float(A.diagonal().sum()) / n,
        max_abs / (mean_abs + 1e-12),
    ], dtype=np.float32)


def sparsity_image(A: sp.csr_matrix, size: int = IMAGE_SIZE) -> np.ndarray:
    """Render the sparsity pattern of A as a binary (size, size) float32 image."""
    coo  = A.tocoo()
    rows = np.minimum((coo.row * size // A.shape[0]).astype(int), size - 1)
    cols = np.minimum((coo.col * size // A.shape[1]).astype(int), size - 1)
    img  = np.zeros((size, size), dtype=np.float32)
    img[rows, cols] = 1.0
    return img


# ── model ─────────────────────────────────────────────────────────────────────

class SolverSelectorNet(nn.Module):
    """
    Dual-branch classifier for iterative solver selection.

    CNN branch  — processes the binary sparsity-pattern image (1 × H × W).
    Stats branch — processes scalar matrix statistics (N_FEATURES,).

    Both embeddings are concatenated and passed to a classification head
    that outputs logits over SOLVERS.
    """

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_SOLVERS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                          # H/2 × W/2
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                          # H/4 × W/4
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                  # → 4 × 4
            nn.Flatten(),                                             # → 1 024
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
        )

        self.stats = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(img), self.stats(feat)], dim=1))


# ── checkpoint helpers ────────────────────────────────────────────────────────

def latest_checkpoint() -> "str | None":
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def load_checkpoint(device: torch.device) -> "tuple[SolverSelectorNet, dict]":
    path = latest_checkpoint()
    if path is None:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    ckpt  = torch.load(path, map_location=device)
    model = SolverSelectorNet(ckpt["n_features"], ckpt["n_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
