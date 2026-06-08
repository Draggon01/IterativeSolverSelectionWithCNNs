"""
mm_model.py — MM-AutoSolver architecture and feature extraction.

Re-implementation of Xiong et al. (2025),
"MM-AutoSolver: A multimodal machine learning method for the
auto-selection of iterative solvers and preconditioners",
Journal of Parallel and Distributed Computing 205, 105144.

Architecture (from paper):
  MLP branch  : BN → FC(1024) → ReLU → BN → FC(128) → ReLU → FC(n_classes)
  CNN branch  : Conv(3×3,Tanh)+MaxPool → Conv(5×5,Tanh)+MaxPool → Flatten → FC(n_classes)
  Fusion      : element-wise addition  (paper eq. 7; NOT concatenation)
  Prediction  : FC(n_classes → n_classes) → logits
  Training    : Adam lr=1e-3, 256 epochs, batch=512, CrossEntropy
"""

import glob
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

MM_N_FEATURES = 17
MM_IMAGE_SIZE  = 128   # paper uses 128×128 density representation

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints")


# ── solver set (paper Table 3 — 19 combinations) ──────────────────────────────
# PETSc type strings for the paper's exact solver-preconditioner pairs.
# "fbcgs" in the paper → PETSc "fbcgsr" (Flexible BiCGSTAB-R).

MM_SOLVER_PAIRS: list[tuple[str, str]] = [
    ("fbcgsr", "jacobi"),    # fbcgs+jacobi  (2,173 train samples in paper)
    ("bcgsl",  "none"),      # bcgsl+none    (2,054)
    ("symmlq", "icc"),       # symmlq+icc    (1,201)  — SPD only
    ("symmlq", "jacobi"),    # symmlq+jacobi   (923)
    ("dgmres", "none"),      # dgmres+none     (650)
    ("gmres",  "gamg"),      # gmres+gamg      (640)
    ("cr",     "eisenstat"), # cr+eisenstat    (598)  — symmetric only
    ("symmlq", "sor"),       # symmlq+sor      (582)
    ("fbcgsr", "ilu"),       # fbcgs+ilu       (562)
    ("minres", "gamg"),      # minres+gamg     (524)
    ("fcg",    "gamg"),      # fcg+gamg        (342)  — symmetric only
    ("cr",     "jacobi"),    # cr+jacobi       (310)
    ("cg",     "ilu"),       # cg+ilu          (275)  — SPD only
    ("fgmres", "gamg"),      # fgmres+gamg     (226)
    ("cg",     "eisenstat"), # cg+eisenstat    (224)  — SPD only
    ("cg",     "bjacobi"),   # cg+bjacobi      (193)  — SPD only
    ("cr",     "ilu"),       # cr+ilu           (68)
    ("cgs",    "gamg"),      # cgs+gamg         (49)
    ("bcgsl",  "asm"),       # bcgsl+asm        (29)
]

MM_SOLVER_NAMES: list[str]        = [f"{k}+{p}" for k, p in MM_SOLVER_PAIRS]
MM_SOLVER_IDX:   dict[tuple, int] = {pair: i for i, pair in enumerate(MM_SOLVER_PAIRS)}
MM_N_SOLVERS:    int               = len(MM_SOLVER_PAIRS)  # 19

# Which pairs to benchmark per matrix type.
# Symmetric-only solvers (cg, minres, symmlq, cr, fcg) and ICC preconditioner
# require at least a symmetric (or SPD) matrix.
_SYM_ONLY_KSP = {"cg", "minres", "symmlq", "cr", "fcg"}

_MM_SPD_PAIRS    = MM_SOLVER_PAIRS
# Symmetric non-SPD: allow all symmetric-capable solvers, but exclude cg and icc (SPD-only)
_MM_SYM_PAIRS    = [p for p in MM_SOLVER_PAIRS if p[0] != "cg" and p[1] != "icc"]
_MM_NONSYM_PAIRS = [p for p in MM_SOLVER_PAIRS
                    if p[0] not in _SYM_ONLY_KSP and p[1] != "icc"]

MM_APPLICABLE: dict[str, list[tuple[str, str]]] = {
    "spd":       _MM_SPD_PAIRS,
    "sym":       _MM_SYM_PAIRS,
    "poisson2d": _MM_SPD_PAIRS,
    "poisson3d": _MM_SPD_PAIRS,
    "nonsym":    _MM_NONSYM_PAIRS,
}


# ── feature extraction ────────────────────────────────────────────────────────

def mm_features(A: sp.csr_matrix) -> np.ndarray:
    """
    17 numerical features from Table 2 of the MM-AutoSolver paper.

    Features:
        0   row_num                 — number of rows
        1   nnz                     — number of non-zeros
        2   nnz_ratio               — nnz / n²
        3   nnz_lower               — nnz in strict lower triangle
        4   nnz_upper               — nnz in strict upper triangle
        5   nnz_diagonal            — number of non-zero diagonal entries
        6   ave_nnz_row             — nnz / n
        7   max_nnz_row             — max nnz in any row
        8   arr_nnz_rows            — variance of per-row nnz
        9   max_value               — max |off-diagonal| value
        10  max_value_diagonal      — max |diagonal| value
        11  diagonal_dominance_ratio — fraction of rows where |a_ii| > Σ|a_ij|, j≠i
        12  is_symmetry             — 1 if ||A-Aᵀ||_F/||A||_F < 0.01
        13  pattern_symmetry        — fraction of NZ positions mirrored in Aᵀ
        14  value_symmetry          — 1 − ||A-Aᵀ||_F/||A||_F
        15  row_variability         — log(1 + max row-wise max/min abs ratio)
        16  col_variability         — log(1 + max col-wise max/min abs ratio)
    """
    A   = A.tocsr()
    n   = A.shape[0]
    nnz = A.nnz
    coo = A.tocoo()

    diag_abs   = np.abs(A.diagonal())
    row_counts = np.diff(A.indptr).astype(np.float64)

    # Off-diagonal max
    off_mask  = coo.row != coo.col
    data_off  = np.abs(coo.data[off_mask]) if off_mask.any() else np.array([0.0])
    max_value = float(data_off.max())

    # Diagonal dominance
    row_abs_sum = np.array(np.abs(A).sum(axis=1)).ravel()
    offdiag_sum = row_abs_sum - diag_abs
    diag_dom    = float(np.mean(diag_abs > offdiag_sum))

    # Symmetry
    frob      = sp.linalg.norm(A, "fro")
    sym_score = sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12)
    is_sym    = float(sym_score < 0.01)

    A_bin        = A.copy(); A_bin.data[:] = 1.0
    pattern_sym  = float(A_bin.multiply(A_bin.T.tocsr()).nnz) / (nnz + 1e-12)
    value_sym    = float(1.0 - sym_score)

    # Row / col variability
    data_abs = np.abs(coo.data) if nnz > 0 else np.array([0.0])
    row_max  = np.zeros(n, dtype=np.float64)
    row_min  = np.full(n, np.inf, dtype=np.float64)
    col_max  = np.zeros(n, dtype=np.float64)
    col_min  = np.full(n, np.inf, dtype=np.float64)
    if nnz > 0:
        np.maximum.at(row_max, coo.row, data_abs)
        np.minimum.at(row_min, coo.row, data_abs)
        np.maximum.at(col_max, coo.col, data_abs)
        np.minimum.at(col_min, coo.col, data_abs)

    def _variability(mx, mn):
        valid = mx > 0
        if not valid.any():
            return 0.0
        return float(np.log1p((mx[valid] / (mn[valid] + 1e-14)).max()))

    row_var = _variability(row_max, row_min)
    col_var = _variability(col_max, col_min)

    return np.array([
        float(n), float(nnz), nnz / (n * n),
        float(sp.tril(A, k=-1).nnz),
        float(sp.triu(A, k=1).nnz),
        float(np.sum(diag_abs > 1e-12)),
        float(nnz) / n,
        float(row_counts.max()) if nnz > 0 else 0.0,
        float(row_counts.var()),
        max_value,
        float(diag_abs.max()) if nnz > 0 else 0.0,
        diag_dom,
        is_sym, pattern_sym, value_sym,
        row_var, col_var,
    ], dtype=np.float32)


def mm_density_image(A: sp.csr_matrix, size: int = MM_IMAGE_SIZE) -> np.ndarray:
    """
    128×128 density representation: count NZs per block, normalised to [0, 1].

    The paper converts a sparse matrix to a fixed-size M×M matrix by dividing
    into M×M blocks and counting the non-zero elements in each block.
    """
    coo  = A.tocoo()
    rows = np.minimum((coo.row * size // A.shape[0]).astype(int), size - 1)
    cols = np.minimum((coo.col * size // A.shape[1]).astype(int), size - 1)
    img  = np.zeros((size, size), dtype=np.float32)
    np.add.at(img, (rows, cols), 1.0)
    mx = img.max()
    if mx > 0:
        img /= mx
    return img


# ── model ─────────────────────────────────────────────────────────────────────

class MMAutoSolverNet(nn.Module):
    """
    Re-implementation of MM-AutoSolver (Xiong et al., 2025).

    The paper's key design choices:
      - MLP with BN before each hidden layer (two hidden layers: 1024, 128)
      - CNN with two conv layers (3×3 then 5×5) + MaxPool, Tanh activation
      - Fusion via element-wise addition (both branches project to n_classes)
      - Final FC prediction head on top of fused embedding

    Forward:
        img  — (B, 1, MM_IMAGE_SIZE, MM_IMAGE_SIZE)
        feat — (B, MM_N_FEATURES)
    Returns:
        logits — (B, n_classes)
    """

    def __init__(self, n_features: int = MM_N_FEATURES, n_classes: int = MM_N_SOLVERS):
        super().__init__()

        # MLP branch: BN → FC(1024) → ReLU → BN → FC(128) → ReLU → FC(n_classes)
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 1024), nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

        # CNN branch: Conv(3×3)+Tanh+MaxPool → Conv(5×5)+Tanh+MaxPool → Flatten → FC(n_classes)
        # With same padding: 128→64→32 after two MaxPools; flatten = 64×32×32 = 65536
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.Tanh(),
            nn.MaxPool2d(2),                                          # → 64×64
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.Tanh(),
            nn.MaxPool2d(2),                                          # → 32×32
            nn.Flatten(),                                             # → 65536
            nn.Linear(64 * (MM_IMAGE_SIZE // 4) * (MM_IMAGE_SIZE // 4), n_classes),
        )

        # Prediction head applied to the fused (element-wise added) embedding
        self.pred_head = nn.Linear(n_classes, n_classes)

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        fn = self.mlp(feat)            # (B, n_classes)
        fv = self.cnn(img)             # (B, n_classes)
        return self.pred_head(fn + fv) # element-wise add then predict


# ── checkpoint helpers ────────────────────────────────────────────────────────

def latest_checkpoint(ckpt_dir: str = CHECKPOINT_DIR) -> "str | None":
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    ckpt_dir: str = CHECKPOINT_DIR,
    device: "torch.device | None" = None,
) -> "tuple[MMAutoSolverNet, dict]":
    if device is None:
        device = torch.device("cpu")
    path = latest_checkpoint(ckpt_dir)
    if path is None:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    ckpt  = torch.load(path, map_location=device)
    model = MMAutoSolverNet(ckpt["n_features"], ckpt["n_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
