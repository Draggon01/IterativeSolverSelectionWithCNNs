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

# Every (KSP type, PC type) pair tried during benchmarking.
# CG/MINRES use ICC (symmetric incomplete Cholesky); general solvers use ILU.
# "bcgs" is PETSc's internal name for BiCGSTAB.
SOLVER_PAIRS: list[tuple[str, str]] = [
    # CG variants — SPD only
    ("cg",     "none"),   ("cg",     "jacobi"), ("cg",     "icc"),
    ("cg",     "sor"),    ("cg",     "gamg"),
    # MINRES variants — symmetric (incl. SPD)
    ("minres", "none"),   ("minres", "jacobi"), ("minres", "icc"),
    ("minres", "sor"),    ("minres", "gamg"),
    # GMRES variants — general
    ("gmres",  "none"),   ("gmres",  "jacobi"), ("gmres",  "ilu"),
    ("gmres",  "sor"),    ("gmres",  "gamg"),
    # BiCG variants — general
    ("bicg",   "none"),   ("bicg",   "jacobi"), ("bicg",   "ilu"),
    ("bicg",   "sor"),    ("bicg",   "gamg"),
    # BiCGSTAB variants — general
    ("bcgs",   "none"),   ("bcgs",   "jacobi"), ("bcgs",   "ilu"),
    ("bcgs",   "sor"),    ("bcgs",   "gamg"),
    # TFQMR variants — general
    ("tfqmr",  "none"),   ("tfqmr",  "jacobi"), ("tfqmr",  "ilu"),
    ("tfqmr",  "sor"),    ("tfqmr",  "gamg"),
]

SOLVER_NAMES: list[str]        = [f"{k}+{p}" for k, p in SOLVER_PAIRS]
SOLVER_IDX:   dict[tuple, int] = {pair: i for i, pair in enumerate(SOLVER_PAIRS)}
N_SOLVERS:    int               = len(SOLVER_PAIRS)   # 30

SOLVERS = SOLVER_NAMES   # alias kept for HDF5 readers and display code

N_FEATURES = 14
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE",  "64"))
IMAGE_MODE  = os.getenv("IMAGE_MODE", "binary")   # binary | density | log_density | magnitude

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints")


# ── feature extraction ────────────────────────────────────────────────────────

def matrix_features(A: sp.csr_matrix) -> np.ndarray:
    """
    Return an (N_FEATURES,) float32 array of scalar matrix statistics.

    Features:
        0   log(1 + n)                  — log-scaled matrix size
        1   log(1 + nnz)                — log-scaled non-zero count
        2   nnz / n²                    — fill ratio (density)
        3   ‖A − Aᵀ‖_F / ‖A‖_F          — 0 = symmetric, ~1 = fully asymmetric
        4   mean(|diag| / Σ|row|)       — diagonal dominance ratio (clipped)
        5   ‖A‖_F / n                   — size-normalised Frobenius norm
        6   tr(A) / n                   — size-normalised trace
        7   max|a_ij| / mean|a_ij|      — relative maximum entry magnitude
        8   log(1 + spectral_radius)    — spectral radius estimate (power iteration)
        9   log(1 + cond_diag)          — diagonal condition proxy max/min |diag|
        10  bandwidth / n               — normalised max distance from main diagonal
        11  diag_nnz_fraction           — fraction of non-zero diagonal entries
        12  row_norm_cv (clipped)       — coefficient of variation of row norms
        13  offdiag_frob_fraction       — ‖A − D‖_F / ‖A‖_F  (off-diagonal energy)
    """
    n   = A.shape[0]
    nnz = A.nnz

    frob     = sp.linalg.norm(A, "fro")
    sym_norm = sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12)

    diag_vals = A.diagonal()
    diag_abs  = np.abs(diag_vals)
    offsum    = np.array(np.abs(A).sum(axis=1)).ravel() - diag_abs
    dom       = float(np.mean(diag_abs / (offsum + 1e-12)))

    vals     = A.data if nnz > 0 else np.array([0.0])
    max_abs  = float(np.abs(vals).max())
    mean_abs = float(np.abs(vals).mean())

    # Spectral radius estimate via 8 power iterations (cheap, O(nnz))
    rng = np.random.default_rng(0)
    v   = rng.standard_normal(n)
    v  /= np.linalg.norm(v) + 1e-12
    for _ in range(8):
        w  = A @ v
        nw = float(np.linalg.norm(w))
        if nw < 1e-12:
            break
        v = w / nw
    spectral_rad = float(abs(v @ (A @ v))) / (float(v @ v) + 1e-12)

    # Diagonal condition proxy: max / min of non-zero |diag| entries
    diag_nz  = diag_abs[diag_abs > 1e-14]
    cond_est = float(diag_nz.max() / diag_nz.min()) if len(diag_nz) >= 2 else 1.0

    # Bandwidth: max |i − j| over all non-zeros, normalised by n
    coo       = A.tocoo()
    bandwidth = (float(np.abs(coo.row.astype(np.int64) - coo.col.astype(np.int64)).max()) / n
                 if nnz > 0 else 0.0)

    # Fraction of diagonal entries that are non-zero
    diag_nnz_frac = float(np.sum(diag_abs > 1e-12)) / n

    # Coefficient of variation of row norms
    row_norms   = np.sqrt(np.array(A.power(2).sum(axis=1)).ravel())
    row_norm_cv = float(row_norms.std() / (row_norms.mean() + 1e-12))

    # Off-diagonal Frobenius fraction
    offdiag_frob = float(sp.linalg.norm(A - sp.diags(diag_vals), "fro")) / (frob + 1e-12)

    return np.array([
        np.log1p(n),
        np.log1p(nnz),
        nnz / (n * n),
        float(sym_norm),
        float(np.clip(dom, 0.0, 20.0)),
        frob / (n + 1e-12),
        float(diag_vals.sum()) / n,
        max_abs / (mean_abs + 1e-12),
        np.log1p(spectral_rad),
        np.log1p(cond_est),
        bandwidth,
        diag_nnz_frac,
        float(np.clip(row_norm_cv, 0.0, 20.0)),
        offdiag_frob,
    ], dtype=np.float32)


def sparsity_image(
    A:    sp.csr_matrix,
    size: int = IMAGE_SIZE,
    mode: str = IMAGE_MODE,
) -> np.ndarray:
    """
    Render the sparsity pattern of A as a (size, size) float32 image.

    mode="binary"      — 1 where any non-zero falls in that cell, else 0
    mode="density"     — non-zero count per cell divided by block area (∈ [0,1])
    mode="log_density" — log(1 + count) per cell, normalised to [0,1]
    mode="magnitude"   — mean |value| per cell, normalised to [0,1]
    """
    coo  = A.tocoo()
    rows = np.minimum((coo.row * size // A.shape[0]).astype(int), size - 1)
    cols = np.minimum((coo.col * size // A.shape[1]).astype(int), size - 1)

    if mode == "binary":
        img = np.zeros((size, size), dtype=np.float32)
        img[rows, cols] = 1.0

    elif mode == "density":
        block_area = (A.shape[0] / size) * (A.shape[1] / size)
        img = np.zeros((size, size), dtype=np.float32)
        np.add.at(img, (rows, cols), 1.0)
        img = np.clip(img / max(block_area, 1.0), 0.0, 1.0)

    elif mode == "log_density":
        img = np.zeros((size, size), dtype=np.float32)
        np.add.at(img, (rows, cols), 1.0)
        img = np.log1p(img)
        mx = img.max()
        if mx > 0:
            img /= mx

    elif mode == "magnitude":
        img   = np.zeros((size, size), dtype=np.float32)
        count = np.zeros((size, size), dtype=np.float32)
        np.add.at(img,   (rows, cols), np.abs(coo.data).astype(np.float32))
        np.add.at(count, (rows, cols), 1.0)
        mask = count > 0
        img[mask] /= count[mask]
        mx = img.max()
        if mx > 0:
            img /= mx

    else:
        raise ValueError(
            f"Unknown IMAGE_MODE {mode!r}. Choose: binary | density | log_density | magnitude"
        )

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
