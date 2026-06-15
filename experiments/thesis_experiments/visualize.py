"""
visualize.py — Graphical analysis of the dataset and model predictions.

Produces four figures saved to $VIZ_DIR:
  01_dataset_overview.png    — label distribution and matrix statistics
  02_sparsity_gallery.png    — example sparsity patterns coloured by best solver
  03_feature_distributions.png — box plots of every feature split by best solver
  04_predictions.png         — confusion matrix, per-class accuracy, probability
                               heatmap  (only when a checkpoint exists)

Environment variables:
  DATA_DIR        Path containing dataset.h5      (default /workspace/data)
  CHECKPOINT_DIR  Path containing epoch_*.pt      (default /workspace/checkpoints)
  VIZ_DIR         Output directory for PNGs       (default ./viz)
  N_GALLERY       Matrices shown in gallery       (default 24)
  N_PREDICT       Samples used for predictions    (default 500)
  SHOW            Set to 1 to open figures interactively (default 0)
  DEVICE          cpu / cuda / auto               (default auto)
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
import torch

from model import SOLVER_PAIRS, SOLVER_NAMES, N_SOLVERS, N_FEATURES, load_checkpoint, SolverSelectorNet

SOLVERS = SOLVER_NAMES   # alias for display code

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = os.getenv("DATA_DIR",       "/workspace/data")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints")
VIZ_DIR        = os.getenv("VIZ_DIR",        "./viz")
N_GALLERY      = int(os.getenv("N_GALLERY",  "24"))
N_PREDICT      = int(os.getenv("N_PREDICT",  "500"))
SHOW           = os.getenv("SHOW", "0") == "1"
_dev           = os.getenv("DEVICE", "auto")
DEVICE         = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)

# Color by KSP type so CG variants, GMRES variants, etc. share a color.
_KSP_TYPES   = ["cg", "minres", "gmres", "bicg", "bcgs", "tfqmr"]
_KSP_PALETTE = plt.cm.tab10(np.linspace(0, 0.6, len(_KSP_TYPES)))
_KSP_COLOR   = {k: _KSP_PALETTE[i] for i, k in enumerate(_KSP_TYPES)}
SOLVER_COLOR = {name: _KSP_COLOR[ksp] for name, (ksp, _) in zip(SOLVER_NAMES, SOLVER_PAIRS)}

FEATURE_NAMES = [
    "log(n)", "log(nnz)", "density", "symmetry score",
    "diag dominance", "Frobenius/n", "trace/n", "max/mean abs",
    "spectral radius", "log cond. est.", "bandwidth/n",
    "diag nnz frac", "row norm CV", "offdiag Frob frac",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        images   = f["images"][:]    # (N, H, W)  float32
        features = f["features"][:] # (N, F)     float32
        labels   = f["labels"][:]   # (N,)       int32
    log.info("Loaded %d samples from %s", len(labels), h5_path)
    return images, features, labels


def save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = os.path.join(VIZ_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved %s", path)
    if SHOW:
        plt.show()
    plt.close(fig)


def solver_legend(fig: plt.Figure, y: float = 0.01) -> None:
    # One patch per KSP type (6 entries) keeps the legend compact.
    patches = [mpatches.Patch(color=_KSP_COLOR[k], label=k) for k in _KSP_TYPES]
    fig.legend(handles=patches, loc="lower center", ncol=len(_KSP_TYPES),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, y),
               title="KSP type (colour group)")


# ── figure 1: dataset overview ────────────────────────────────────────────────

def fig_dataset_overview(
    images: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Dataset Overview", fontsize=16, fontweight="bold")

    # Label distribution
    ax = axes[0, 0]
    counts = np.bincount(labels, minlength=N_SOLVERS)
    bars   = ax.bar(range(N_SOLVERS), counts, color=[SOLVER_COLOR[s] for s in SOLVERS])
    ax.bar_label(bars, fontsize=6)
    ax.set_title("Solver Label Distribution")
    ax.set_xlabel("Best solver"); ax.set_ylabel("Count")
    ax.set_xticks(range(N_SOLVERS))
    ax.set_xticklabels(SOLVERS, rotation=90, ha="center", fontsize=6)

    # Matrix size distribution — group by KSP type for readability
    ax = axes[0, 1]
    for ksp in _KSP_TYPES:
        mask = np.array([SOLVER_PAIRS[int(l)][0] == ksp for l in labels])
        if mask.any():
            ax.hist(np.expm1(features[mask, 0]), bins=30, alpha=0.55,
                    label=ksp, color=_KSP_COLOR[ksp])
    ax.set_title("Matrix Size Distribution")
    ax.set_xlabel("n  (matrix dimension)"); ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # Density distribution — group by KSP type
    ax = axes[0, 2]
    for ksp in _KSP_TYPES:
        mask = np.array([SOLVER_PAIRS[int(l)][0] == ksp for l in labels])
        if mask.any():
            ax.hist(features[mask, 2], bins=30, alpha=0.55,
                    label=ksp, color=_KSP_COLOR[ksp])
    ax.set_title("Fill Ratio (Density)")
    ax.set_xlabel("nnz / n²"); ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    def _boxplot(ax, feat_col, title, ylabel):
        bp = ax.boxplot(
            [features[labels == i, feat_col] for i in range(N_SOLVERS)],
            patch_artist=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, s in zip(bp["boxes"], SOLVERS):
            patch.set_facecolor(SOLVER_COLOR[s]); patch.set_alpha(0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, N_SOLVERS + 1))
        ax.set_xticklabels(SOLVERS, rotation=90, ha="center", fontsize=5)
        ax.grid(axis="y", alpha=0.3)

    _boxplot(axes[1, 0], 3, "Symmetry Score by Best Solver",
             "0 = symmetric  →  1 = asymmetric")
    _boxplot(axes[1, 1], 4, "Diagonal Dominance by Best Solver",
             "mean( |diag| / Σ|row| )")
    _boxplot(axes[1, 2], 5, "Normalised Frobenius Norm by Best Solver",
             "‖A‖_F / n")

    plt.tight_layout()
    return fig


# ── figure 2: sparsity pattern gallery ────────────────────────────────────────

def fig_sparsity_gallery(
    images: np.ndarray,
    labels: np.ndarray,
    n: int = 24,
) -> plt.Figure:
    """Show n sparsity patterns balanced across solver classes."""
    rng = np.random.default_rng(0)
    per_class = max(1, n // N_SOLVERS)
    indices: list[int] = []
    for i in range(N_SOLVERS):
        pool = np.where(labels == i)[0]
        if len(pool):
            indices.extend(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    indices = indices[:n]
    rng.shuffle(indices)

    cols = 6
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    fig.suptitle(
        "Sparsity Pattern Gallery  (border + title colour = best solver)",
        fontsize=14, fontweight="bold",
    )
    axes = np.array(axes).ravel()

    for ax, idx in zip(axes, indices):
        solver = SOLVERS[labels[idx]]
        color  = SOLVER_COLOR[solver]
        ax.imshow(images[idx], cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(solver, fontsize=9, color="white",
                     bbox=dict(facecolor=color, pad=2, edgecolor="none", boxstyle="round"))
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    solver_legend(fig, y=0.0)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ── figure 3: feature distributions ──────────────────────────────────────────

def fig_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    fig.suptitle("Feature Distributions by Best Solver", fontsize=14, fontweight="bold")

    for fi, (ax, name) in enumerate(zip(axes.ravel(), FEATURE_NAMES)):
        bp = ax.boxplot(
            [features[labels == i, fi] for i in range(N_SOLVERS)],
            patch_artist=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        for patch, s in zip(bp["boxes"], SOLVERS):
            patch.set_facecolor(SOLVER_COLOR[s]); patch.set_alpha(0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(1, N_SOLVERS + 1))
        ax.set_xticklabels(SOLVERS, rotation=90, ha="center", fontsize=5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


# ── figure 4: prediction analysis ────────────────────────────────────────────

def fig_predictions(
    images: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    model: SolverSelectorNet,
    n: int = 500,
) -> plt.Figure:
    rng = np.random.default_rng(1)
    n   = min(n, len(labels))
    idx = rng.choice(len(labels), n, replace=False)

    # Run inference in one batch (< 10 MB for n=500 with 64×64 images)
    img_t  = torch.from_numpy(images[idx][:, None]).to(DEVICE)
    feat_t = torch.from_numpy(features[idx]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(img_t, feat_t), dim=1).cpu().numpy()
    preds  = probs.argmax(axis=1)
    true   = labels[idx]
    acc    = (preds == true).mean()

    # Confusion matrix
    cm = np.zeros((N_SOLVERS, N_SOLVERS), dtype=int)
    for t, p in zip(true, preds):
        cm[t, p] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Model Predictions  (n={n} samples  |  overall accuracy={acc:.2%})",
        fontsize=14, fontweight="bold",
    )

    # [0] Confusion matrix
    ax  = axes[0]
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_SOLVERS)); ax.set_xticklabels(SOLVERS, rotation=45, ha="right")
    ax.set_yticks(range(N_SOLVERS)); ax.set_yticklabels(SOLVERS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() * 0.5
    for i in range(N_SOLVERS):
        for j in range(N_SOLVERS):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)

    # [1] Per-solver accuracy
    ax = axes[1]
    per_acc = np.array([
        cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        for i in range(N_SOLVERS)
    ])
    bars = ax.bar(range(N_SOLVERS), per_acc, color=[SOLVER_COLOR[s] for s in SOLVERS])
    ax.bar_label(bars, fmt="%.2f", fontsize=6)
    ax.axhline(acc, color="black", linestyle="--", linewidth=1.2, label=f"overall {acc:.2%}")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Solver Accuracy")
    ax.set_xlabel("Solver"); ax.set_ylabel("Accuracy")
    ax.set_xticks(range(N_SOLVERS))
    ax.set_xticklabels(SOLVERS, rotation=90, ha="center", fontsize=5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # [2] Predicted probability heatmap for the first 20 samples
    ax      = axes[2]
    n_show  = min(20, n)
    s_probs = probs[:n_show]
    s_true  = true[:n_show]
    s_pred  = preds[:n_show]
    im2     = ax.imshow(s_probs, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(N_SOLVERS))
    ax.set_xticklabels(SOLVERS, rotation=90, ha="center", fontsize=5)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(
        [f"{SOLVERS[t]} → {SOLVERS[p]}" for t, p in zip(s_true, s_pred)],
        fontsize=6,
    )
    ax.set_title("Predicted Probabilities\n(first 20 samples)")
    ax.set_xlabel("Solver")
    fig.colorbar(im2, ax=ax, label="probability")

    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    if not os.path.exists(h5_path):
        log.error("Dataset not found at %s — run generate_data.py first.", h5_path)
        return

    images, features, labels = load_data(h5_path)

    save_fig(fig_dataset_overview(images, features, labels),  "01_dataset_overview")
    save_fig(fig_sparsity_gallery(images, labels, N_GALLERY), "02_sparsity_gallery")
    save_fig(fig_feature_distributions(features, labels),     "03_feature_distributions")

    try:
        model, ckpt = load_checkpoint(DEVICE)
        log.info(
            "Checkpoint found (epoch=%d  val_acc=%.4f) — generating prediction figure.",
            ckpt.get("epoch", -1), ckpt.get("val_acc", float("nan")),
        )
        save_fig(fig_predictions(images, features, labels, model, N_PREDICT), "04_predictions")
    except FileNotFoundError:
        log.info("No checkpoint found in %s — skipping prediction figure.", CHECKPOINT_DIR)

    log.info("All figures saved to %s/", VIZ_DIR)


if __name__ == "__main__":
    main()
