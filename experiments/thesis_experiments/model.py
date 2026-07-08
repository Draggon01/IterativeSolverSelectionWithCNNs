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

# ── Solver set A — original 19 pairs (MM-AutoSolver compatible) ───────────────
_SOLVER_PAIRS_MAIN: list[tuple[str, str]] = [
    ("fbcgsr", "jacobi"),    # fbcgsr+jacobi
    ("bcgsl",  "none"),      # bcgsl+none
    ("symmlq", "icc"),       # symmlq+icc      — SPD only
    ("symmlq", "jacobi"),    # symmlq+jacobi   — symmetric only
    ("dgmres", "none"),      # dgmres+none
    ("gmres",  "gamg"),      # gmres+gamg
    ("cr",     "eisenstat"), # cr+eisenstat    — symmetric only
    ("symmlq", "sor"),       # symmlq+sor      — symmetric only
    ("fbcgsr", "ilu"),       # fbcgsr+ilu
    ("minres", "gamg"),      # minres+gamg     — symmetric only
    ("fcg",    "gamg"),      # fcg+gamg        — symmetric only
    ("cr",     "jacobi"),    # cr+jacobi       — symmetric only
    ("cg",     "ilu"),       # cg+ilu          — SPD only
    ("fgmres", "gamg"),      # fgmres+gamg
    ("cg",     "eisenstat"), # cg+eisenstat    — SPD only
    ("cg",     "bjacobi"),   # cg+bjacobi      — SPD only
    ("cr",     "ilu"),       # cr+ilu          — symmetric only
    ("cgs",    "gamg"),      # cgs+gamg
    ("bcgsl",  "asm"),       # bcgsl+asm
]
_SYM_ONLY_KSP_MAIN = {"cg", "minres", "symmlq", "cr", "fcg"}
_APPLICABLE_MAIN: dict[str, list[tuple[str, str]]] = {
    "spd":       _SOLVER_PAIRS_MAIN,
    "sym":       [p for p in _SOLVER_PAIRS_MAIN if p[0] != "cg" and p[1] != "icc"],
    "poisson2d": _SOLVER_PAIRS_MAIN,
    "poisson3d": _SOLVER_PAIRS_MAIN,
    "nonsym":    [p for p in _SOLVER_PAIRS_MAIN
                  if p[0] not in _SYM_ONLY_KSP_MAIN and p[1] != "icc"],
}

# ── Solver set B — alternative 15 pairs (practical workhorse combinations) ────
# Select with SOLVER_SET=alt
_SOLVER_PAIRS_ALT: list[tuple[str, str]] = [
    ("cg",     "gamg"),      # cg+gamg         — SPD only
    ("cg",     "none"),      # cg+none          — SPD only
    ("gmres",  "ilu"),       # gmres+ilu
    ("gmres",  "bjacobi"),   # gmres+bjacobi
    ("gmres",  "none"),      # gmres+none
    ("fgmres", "ilu"),       # fgmres+ilu
    ("fgmres", "bjacobi"),   # fgmres+bjacobi
    ("bcgs",   "ilu"),       # bcgs+ilu
    ("bcgs",   "jacobi"),    # bcgs+jacobi
    ("bcgs",   "gamg"),      # bcgs+gamg
    ("bcgs",   "none"),      # bcgs+none
    ("tfqmr",  "ilu"),       # tfqmr+ilu
    ("tfqmr",  "jacobi"),    # tfqmr+jacobi
    ("lgmres", "ilu"),       # lgmres+ilu
    ("gcr",    "ilu"),       # gcr+ilu
]
_SYM_ONLY_KSP_ALT = {"cg"}
_APPLICABLE_ALT: dict[str, list[tuple[str, str]]] = {
    "spd":       _SOLVER_PAIRS_ALT,
    "sym":       [p for p in _SOLVER_PAIRS_ALT if p[0] != "cg"],
    "poisson2d": _SOLVER_PAIRS_ALT,
    "poisson3d": _SOLVER_PAIRS_ALT,
    "nonsym":    [p for p in _SOLVER_PAIRS_ALT if p[0] != "cg"],
}

# ── Active solver set — controlled by SOLVER_SET env var ──────────────────────
# SOLVER_SET=main  (default) → set A, 19 pairs, backward-compatible
# SOLVER_SET=alt             → set B, 15 pairs, new experiments
_SOLVER_SET = os.getenv("SOLVER_SET", "main").strip().lower()
if _SOLVER_SET == "alt":
    SOLVER_PAIRS  = _SOLVER_PAIRS_ALT
    APPLICABLE    = _APPLICABLE_ALT
else:
    SOLVER_PAIRS  = _SOLVER_PAIRS_MAIN
    APPLICABLE    = _APPLICABLE_MAIN

SOLVER_NAMES: list[str]        = [f"{k}+{p}" for k, p in SOLVER_PAIRS]
SOLVER_IDX:   dict[tuple, int] = {pair: i for i, pair in enumerate(SOLVER_PAIRS)}
N_SOLVERS:    int               = len(SOLVER_PAIRS)   # 19

SOLVERS = SOLVER_NAMES   # alias kept for HDF5 readers and display code

N_FEATURES = 20
FEATURE_NAMES: list[str] = [
    "log(1+n)",
    "log(1+nnz)",
    "nnz/n²  (density)",
    "‖A−Aᵀ‖_F/‖A‖_F  (asymmetry)",
    "mean(|diag|/Σ|row|)  (diag dominance mean)",
    "‖A‖_F/n  (norm/size)",
    "tr(A)/n  (trace/size)",
    "max|a_ij|/mean|a_ij|  (rel max entry)",
    "log(1+spectral_radius)",
    "log(1+cond_diag)  (diag cond proxy)",
    "bandwidth/n  (norm bandwidth)",
    "diag_nnz_fraction",
    "row_norm_cv  (row norm variation)",
    "offdiag_frob_fraction  (‖A−D‖_F/‖A‖_F)",
    "neg_offdiag_frac  (fraction negative off-diag entries)",
    "pos_diag_frac  (fraction strictly positive diagonal entries)",
    "struct_sym_frac  (structural symmetry fraction)",
    "min_diag_dominance  (min |d_i|/Σ|off_i|, clipped)",
    "nnz_per_row_cv  (CV of per-row nnz counts)",
    "diag_dom_variance  (variance of per-row dominance ratios, clipped)",
]
IMAGE_SIZE  = int(os.getenv("IMAGE_SIZE",  "64"))
IMAGE_MODE  = os.getenv("IMAGE_MODE", "binary")
IMAGE_MODE2 = os.getenv("IMAGE_MODE2", "")        # empty = single-channel
NO_CNN      = os.getenv("NO_CNN", "0") == "1"
MODEL_SIZE  = os.getenv("MODEL_SIZE", "small")     # small | large

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
        4   mean(|diag| / Σ|row|)       — diagonal dominance mean (clipped)
        5   ‖A‖_F / n                   — size-normalised Frobenius norm
        6   tr(A) / n                   — size-normalised trace
        7   max|a_ij| / mean|a_ij|      — relative maximum entry magnitude
        8   log(1 + spectral_radius)    — spectral radius estimate (power iteration)
        9   log(1 + cond_diag)          — diagonal condition proxy max/min |diag|
        10  bandwidth / n               — normalised max distance from main diagonal
        11  diag_nnz_fraction           — fraction of non-zero diagonal entries
        12  row_norm_cv (clipped)       — coefficient of variation of row norms
        13  offdiag_frob_fraction       — ‖A − D‖_F / ‖A‖_F  (off-diagonal energy)
        14  neg_offdiag_frac            — fraction of off-diagonal entries that are negative
                                          ~1 for M-matrices (SPD-like); ~0.5 for indefinite
        15  pos_diag_frac               — fraction of strictly positive diagonal entries
                                          1.0 for SPD; <1 for indefinite / singular-like
        16  struct_sym_frac             — fraction of (i,j) non-zeros with matching (j,i)
                                          distinguishes structurally symmetric from asymmetric
        17  min_diag_dominance          — min_i(|d_i| / Σ|off_i|) (clipped)
                                          detects worst-case row for ILU breakdown
        18  nnz_per_row_cv              — CV of per-row non-zero counts (clipped)
                                          low = structured grid; high = unstructured
        19  diag_dom_variance           — variance of per-row dominance ratios (clipped)
                                          patchy dominance predicts ILU instability
    """
    n   = A.shape[0]
    nnz = A.nnz

    frob     = sp.linalg.norm(A, "fro")
    sym_norm = sp.linalg.norm(A - A.T, "fro") / (frob + 1e-12)

    diag_vals = A.diagonal()
    diag_abs  = np.abs(diag_vals)
    offsum    = np.array(np.abs(A).sum(axis=1)).ravel() - diag_abs
    dom_ratios = diag_abs / (offsum + 1e-12)
    dom        = float(np.mean(dom_ratios))

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

    # ── new features (14–19) ──────────────────────────────────────────────────

    # 14: fraction of off-diagonal entries that are negative
    if nnz > 0:
        offdiag_mask = coo.row != coo.col
        offdiag_data = coo.data[offdiag_mask]
        neg_offdiag_frac = (float(np.sum(offdiag_data < 0)) / len(offdiag_data)
                            if len(offdiag_data) > 0 else 0.5)
    else:
        neg_offdiag_frac = 0.5

    # 15: fraction of strictly positive diagonal entries
    pos_diag_frac = float(np.sum(diag_vals > 1e-14)) / n

    # 16: structural symmetry fraction — (i,j) entries that have a matching (j,i)
    if nnz > 0:
        pairs     = set(zip(coo.row.tolist(), coo.col.tolist()))
        n_sym     = sum(1 for (r, c) in pairs if r != c and (c, r) in pairs)
        n_offdiag = sum(1 for (r, c) in pairs if r != c)
        struct_sym_frac = float(n_sym) / (n_offdiag + 1e-12)
    else:
        struct_sym_frac = 1.0

    # 17: minimum per-row diagonal dominance ratio
    min_diag_dom = float(np.clip(dom_ratios.min(), 0.0, 20.0))

    # 18: CV of per-row non-zero counts
    nnz_per_row    = np.diff(A.indptr).astype(np.float64)
    nnz_per_row_cv = float(np.clip(
        nnz_per_row.std() / (nnz_per_row.mean() + 1e-12), 0.0, 20.0
    ))

    # 19: variance of per-row diagonal dominance ratios
    diag_dom_var = float(np.clip(dom_ratios.var(), 0.0, 20.0))

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
        neg_offdiag_frac,
        pos_diag_frac,
        struct_sym_frac,
        min_diag_dom,
        nnz_per_row_cv,
        diag_dom_var,
    ], dtype=np.float32)


def sparsity_image(
    A:    sp.csr_matrix,
    size: int = IMAGE_SIZE,
    mode: str = IMAGE_MODE,
) -> np.ndarray:
    """
    Render the sparsity pattern of A as a (size, size) float32 image.

    mode="binary"           — 1 where any non-zero falls in that cell, else 0
    mode="density"          — non-zero count per cell / block area (∈ [0,1])
    mode="log_density"      — log(1 + count) per cell, normalised to [0,1]
    mode="magnitude"        — mean |value| per cell, normalised to [0,1]
    mode="symmetry"         — |A − Aᵀ| per cell, normalised to [0,1]
    mode="diagonal"         — off-diag = 0.5, diagonal entries = 1.0
    mode="sign"             — mean sign per cell: positive→1.0, negative→0.0,
                              empty→0.5. Directly shows positive/negative structure
                              and diagonal sign pattern (SPD vs indefinite).
    mode="signed_magnitude" — sign(a)·log(1+|a|) per cell, averaged, normalised
                              to [0,1] with 0.5=zero. Combines magnitude and sign:
                              strictly more informative than either alone.
    mode="rcm_<base>"       — apply Reverse Cuthill-McKee reordering first,
                              then render with <base> mode (e.g. rcm_magnitude).
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

    elif mode == "symmetry":
        # |A − Aᵀ| per cell — highlights asymmetric entries, zero for symmetric ones
        sym_diff = np.abs(A - A.T).tocoo()
        sr = np.minimum((sym_diff.row * size // A.shape[0]).astype(int), size - 1)
        sc = np.minimum((sym_diff.col * size // A.shape[1]).astype(int), size - 1)
        img = np.zeros((size, size), dtype=np.float32)
        np.add.at(img, (sr, sc), np.abs(sym_diff.data).astype(np.float32))
        mx = img.max()
        if mx > 0:
            img /= mx

    elif mode == "diagonal":
        # Off-diagonal non-zeros = 0.5, diagonal entries = 1.0
        # Gives the CNN explicit information about diagonal structure
        img = np.zeros((size, size), dtype=np.float32)
        img[rows, cols] = 0.5
        is_diag = coo.row == coo.col
        img[rows[is_diag], cols[is_diag]] = 1.0

    elif mode == "sign":
        # Mean sign per cell, mapped from [-1, 1] → [0, 1]; empty cells = 0.5.
        img = np.full((size, size), 0.5, dtype=np.float32)
        if A.nnz > 0:
            raw = np.zeros((size, size), dtype=np.float32)
            cnt = np.zeros((size, size), dtype=np.float32)
            np.add.at(raw, (rows, cols), np.sign(coo.data).astype(np.float32))
            np.add.at(cnt, (rows, cols), 1.0)
            mask = cnt > 0
            img[mask] = (raw[mask] / cnt[mask] + 1.0) / 2.0

    elif mode == "signed_magnitude":
        # sign(a) × log(1 + |a|) per cell, averaged, normalised to [0, 1].
        # 0.5 = empty or zero net; >0.5 = net positive; <0.5 = net negative.
        img = np.full((size, size), 0.5, dtype=np.float32)
        if A.nnz > 0:
            raw = np.zeros((size, size), dtype=np.float32)
            cnt = np.zeros((size, size), dtype=np.float32)
            vals = (np.sign(coo.data) * np.log1p(np.abs(coo.data))).astype(np.float32)
            np.add.at(raw, (rows, cols), vals)
            np.add.at(cnt, (rows, cols), 1.0)
            mask = cnt > 0
            raw[mask] /= cnt[mask]
            mx = float(np.abs(raw[mask]).max()) if mask.any() else 0.0
            if mx > 0:
                img[mask] = raw[mask] / (2.0 * mx) + 0.5

    elif mode.startswith("rcm_"):
        # Apply Reverse Cuthill-McKee reordering then render with the base mode
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        perm = reverse_cuthill_mckee(A, symmetric_mode=False)
        return sparsity_image(A[perm][:, perm], size=size, mode=mode[4:])

    else:
        raise ValueError(
            f"Unknown IMAGE_MODE {mode!r}. Choose: binary | density | log_density | "
            "magnitude | symmetry | diagonal | sign | signed_magnitude | rcm_<mode>"
        )

    return img


# ── model ─────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Two conv layers with a residual skip connection. Input and output channels are equal."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SolverSelectorNet(nn.Module):
    """
    Dual-branch classifier for iterative solver selection.

    CNN branch   — processes the sparsity-pattern image (in_channels × H × W).
                   model_size="small": 2 ResBlocks, 64 channels, ~0.5M params.
                   model_size="large": 3 ResBlocks, 128 channels, ~2M params.
                   Disabled when no_cnn=True.
    Stats branch — processes scalar matrix statistics (n_features,).

    in_channels=2 enables dual-mode input (two image representations stacked).
    """

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_SOLVERS,
                 no_cnn: bool = False, model_size: str = "small", in_channels: int = 1):
        super().__init__()
        self.no_cnn      = no_cnn
        self.model_size  = model_size
        self.in_channels = in_channels

        large   = (model_size == "large")
        cnn_ch  = 128  if large else 64
        cnn_out = 512  if large else 256
        stat_h  = 128  if large else 64
        head_h  = 256  if large else 128
        drop    = 0.4  if large else 0.3

        if not no_cnn:
            layers = [
                nn.Conv2d(in_channels, cnn_ch, 3, padding=1),
                nn.BatchNorm2d(cnn_ch), nn.ReLU(),
                nn.MaxPool2d(2),                       # H/2
                _ResBlock(cnn_ch),
                nn.MaxPool2d(2),                       # H/4
                _ResBlock(cnn_ch),
            ]
            if large:
                layers += [nn.MaxPool2d(2), _ResBlock(cnn_ch)]  # H/8, extra stage
            layers += [
                nn.AdaptiveAvgPool2d(4),               # → 4×4
                nn.Flatten(),                          # → cnn_ch * 16
                nn.Linear(cnn_ch * 16, cnn_out), nn.ReLU(), nn.Dropout(drop),
            ]
            self.cnn = nn.Sequential(*layers)

        self.stats = nn.Sequential(
            nn.Linear(n_features, stat_h), nn.ReLU(),
            nn.Linear(stat_h, stat_h), nn.ReLU(),
        )

        cnn_dim = 0 if no_cnn else cnn_out
        self.head = nn.Sequential(
            nn.Linear(cnn_dim + stat_h, head_h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(head_h, n_classes),
        )

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        stats_out = self.stats(feat)
        if self.no_cnn:
            return self.head(stats_out)
        return self.head(torch.cat([self.cnn(img), stats_out], dim=1))


# ── checkpoint helpers ────────────────────────────────────────────────────────

def latest_checkpoint() -> "str | None":
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def load_checkpoint(device: torch.device) -> "tuple[SolverSelectorNet, dict]":
    path = latest_checkpoint()
    if path is None:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    ckpt  = torch.load(path, map_location=device)
    model = SolverSelectorNet(ckpt["n_features"], ckpt["n_classes"],
                              no_cnn=ckpt.get("no_cnn", False)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
