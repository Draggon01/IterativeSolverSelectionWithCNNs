"""
mm_evaluate.py — Evaluate the MM-AutoSolver baseline on the held-out validation set.

Reports the four metrics used in the paper (Table 5):
  Acc   — overall accuracy
  MP    — macro precision
  MR    — macro recall
  F1    — macro F1 score

Also prints per-class accuracy and label distribution for dataset comparison.

Environment variables:
  DATA_DIR        HDF5 dataset directory  (default /workspace/data)
  CHECKPOINT_DIR  Checkpoint directory    (default /workspace/checkpoints)
  VAL_SPLIT       Validation fraction     (default 0.10)
  DEVICE          cpu | cuda | auto       (default auto)
"""

import logging
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from mm_model import MMAutoSolverNet, MM_N_FEATURES, MM_SOLVER_NAMES, MM_N_SOLVERS, load_checkpoint
from mm_train import MMDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR       = os.getenv("DATA_DIR",       "./data")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints")
VAL_SPLIT      = float(os.getenv("VAL_SPLIT", "0.10"))

_dev   = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)

# Paper's class distribution (Table 3) for comparison
PAPER_COUNTS = {
    "fbcgsr+jacobi": 2173, "bcgsl+none": 2054, "symmlq+icc": 1201,
    "symmlq+jacobi": 923,  "dgmres+none": 650,  "gmres+gamg": 640,
    "cr+eisenstat":  598,  "symmlq+sor":  582,  "fbcgsr+ilu": 562,
    "minres+gamg":   524,  "fcg+gamg":    342,  "cr+jacobi":  310,
    "cg+ilu":        275,  "fgmres+gamg": 226,  "cg+eisenstat": 224,
    "cg+bjacobi":    193,  "cr+ilu":       68,  "cgs+gamg":    49,
    "bcgsl+asm":      29,
}


def near_optimal_accuracy(
    y_pred: np.ndarray,
    runtimes: np.ndarray,
    tolerance: float = 0.10,
) -> float:
    """Fraction of predictions whose runtime is within `tolerance` of the best."""
    best = np.nanmin(runtimes, axis=1, keepdims=True)
    threshold = best * (1.0 + tolerance)
    pred_times = runtimes[np.arange(len(y_pred)), y_pred]
    return float(np.mean(pred_times <= threshold.squeeze()))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    acc = float(np.mean(y_true == y_pred))
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p  = tp / (tp + fp + 1e-12)
        r  = tp / (tp + fn + 1e-12)
        f  = 2 * p * r / (p + r + 1e-12)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {"Acc": acc, "MP": float(np.mean(precisions)),
            "MR": float(np.mean(recalls)), "F1": float(np.mean(f1s)),
            "per_class_p": precisions, "per_class_r": recalls, "per_class_f1": f1s}


def main() -> None:
    dataset = MMDataset(os.path.join(DATA_DIR, "dataset.h5"))
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val

    # Same fixed split as training
    _, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model, ckpt = load_checkpoint(CHECKPOINT_DIR, DEVICE)
    log.info("Checkpoint: epoch=%d  val_acc=%.4f", ckpt["epoch"], ckpt["val_acc"])

    n_classes = ckpt["n_classes"]
    all_preds, all_labels = [], []

    # Collect val indices to load runtimes after inference
    val_indices = list(val_ds.indices) if hasattr(val_ds, "indices") else None

    model.eval()
    with torch.no_grad():
        for img, feat, lbl in val_loader:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            all_preds.append(model(img, feat).argmax(1).cpu().numpy())
            all_labels.append(lbl.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Load runtimes for near-optimal accuracy
    with h5py.File(os.path.join(DATA_DIR, "dataset.h5"), "r") as f:
        if val_indices is not None:
            val_runtimes = f["runtimes"][sorted(val_indices)]
            # reorder to match original val_ds order
            order = np.argsort(np.argsort(val_indices))
            val_runtimes = val_runtimes[order]
        else:
            val_runtimes = f["runtimes"][n_train:]

    metrics = compute_metrics(y_true, y_pred, n_classes)
    noa_5  = near_optimal_accuracy(y_pred, val_runtimes, tolerance=0.05)
    noa_10 = near_optimal_accuracy(y_pred, val_runtimes, tolerance=0.10)
    noa_20 = near_optimal_accuracy(y_pred, val_runtimes, tolerance=0.20)

    # Label distribution in full dataset
    with h5py.File(os.path.join(DATA_DIR, "dataset.h5"), "r") as f:
        all_labels_full = f["labels"][:]
    our_counts = np.bincount(all_labels_full, minlength=n_classes)

    print("\n── MM-AutoSolver Baseline Results ──────────────────────────────")
    print(f"  Checkpoint   : epoch {ckpt['epoch']}")
    print(f"  Val samples  : {len(y_true)}  (of {len(dataset)} total)")
    print(f"  Classes      : {n_classes}")
    print()
    print(f"  Accuracy (Acc) : {metrics['Acc']*100:.2f}%   (paper: 78.54%)")
    print(f"  Macro Precision: {metrics['MP']*100:.2f}%   (paper: 63.41%)")
    print(f"  Macro Recall   : {metrics['MR']*100:.2f}%   (paper: 62.81%)")
    print(f"  Macro F1       : {metrics['F1']*100:.2f}%   (paper: 62.53%)")
    print()
    print(f"  Near-optimal Acc (±5%)  : {noa_5*100:.2f}%")
    print(f"  Near-optimal Acc (±10%) : {noa_10*100:.2f}%")
    print(f"  Near-optimal Acc (±20%) : {noa_20*100:.2f}%")
    print()
    print(f"  {'Solver':<20} {'Our N':>7}  {'Paper N':>7}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    for i, name in enumerate(MM_SOLVER_NAMES):
        paper_n = PAPER_COUNTS.get(name, 0)
        print(f"  {name:<20} {our_counts[i]:>7}  {paper_n:>7}  "
              f"{metrics['per_class_f1'][i]*100:>5.1f}%  "
              f"{metrics['per_class_p'][i]*100:>5.1f}%  "
              f"{metrics['per_class_r'][i]*100:>5.1f}%")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
