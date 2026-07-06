"""
evaluate.py — Evaluate a trained thesis_experiments model on the held-out validation set.

Reports the same four metrics as the MM-AutoSolver paper for direct comparison:
  Acc   — overall accuracy
  MP    — macro precision
  MR    — macro recall
  F1    — macro F1 score

Also reports top-2/top-3 accuracy and near-optimal solver accuracy (within X% of best runtime).

Environment variables:
  EXPERIMENT      Run name (checkpoints loaded from CHECKPOINT_DIR/EXPERIMENT/)
  DATA_DIR        HDF5 dataset directory   (default /workspace/data)
  CHECKPOINT_DIR  Checkpoint root          (default /workspace/checkpoints)
  VAL_SPLIT       Validation fraction      (default 0.10)
  DEVICE          cpu | cuda | auto        (default auto)
"""

import glob
import logging
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import SolverSelectorNet, SOLVER_NAMES, N_SOLVERS
from train import SolverDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EXPERIMENT     = os.getenv("EXPERIMENT",      "default")
DATA_DIR       = os.getenv("DATA_DIR",        "/workspace/data")
_CKPT_ROOT     = os.getenv("CHECKPOINT_DIR",  "/workspace/checkpoints")
CHECKPOINT_DIR = os.path.join(_CKPT_ROOT, EXPERIMENT)
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


def load_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    ckpt  = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    model = SolverSelectorNet(
        ckpt["n_features"], ckpt["n_classes"],
        no_cnn=ckpt.get("no_cnn", False),
        model_size=ckpt.get("model_size", "small"),
        in_channels=ckpt.get("in_channels", 1),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


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


def near_optimal_accuracy(y_pred: np.ndarray, runtimes: np.ndarray, tol: float) -> float:
    best      = np.nanmin(runtimes, axis=1, keepdims=True)
    threshold = best * (1.0 + tol)
    pred_t    = runtimes[np.arange(len(y_pred)), y_pred]
    return float(np.mean(pred_t <= threshold.squeeze()))


def mean_runtime_ratio(y_pred: np.ndarray, runtimes: np.ndarray) -> tuple[float, float]:
    """
    Mean ratio of predicted solver runtime to optimal runtime.
    Returns (mean_ratio, median_ratio). Only counts samples where the
    predicted solver converged; samples where it diverged (NaN runtime)
    are excluded from the ratio but counted in failure_rate instead.
    A value of 1.0 = always picks the best; 2.0 = twice as slow on average.
    """
    best   = np.nanmin(runtimes, axis=1)
    pred_t = runtimes[np.arange(len(y_pred)), y_pred]
    # Only include samples where both the prediction and optimal are finite
    valid  = np.isfinite(pred_t) & np.isfinite(best) & (best > 0)
    ratios = pred_t[valid] / best[valid]
    if valid.sum() == 0:
        return float("nan"), float("nan")
    return float(np.mean(ratios)), float(np.median(ratios))


def failure_rate(y_pred: np.ndarray, runtimes: np.ndarray) -> tuple[float, int]:
    """
    Fraction of predictions that chose a solver which did not converge (NaN runtime).
    Returns (rate, count).
    """
    pred_t   = runtimes[np.arange(len(y_pred)), y_pred]
    n_failed = int(np.sum(~np.isfinite(pred_t)))
    return n_failed / len(y_pred), n_failed


def main() -> None:
    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    dataset = SolverDataset(h5_path)

    # Reproduce the same stratified split used during training
    rng_split = np.random.default_rng(42)
    with h5py.File(h5_path, "r") as f:
        all_labels = f["labels"][:]
    val_idx = []
    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        n_val = max(1, int(len(idx) * VAL_SPLIT))
        val_idx.extend(rng_split.choice(idx, size=n_val, replace=False).tolist())

    val_ds     = Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model, ckpt = load_checkpoint()
    n_classes   = ckpt["n_classes"]
    log.info("Experiment=%s  epoch=%d  val_acc_train=%.4f",
             EXPERIMENT, ckpt["epoch"], ckpt["val_acc"])

    all_preds, all_labels_out, all_topk = [], [], []
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        for img, feat, lbl, _ in val_loader:   # _ = converge_mask, not needed for eval
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            logits = model(img, feat)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels_out.append(lbl.numpy())
            all_topk.append(logits.argsort(dim=1, descending=True)[:, :3].cpu().numpy())
    infer_total_s = time.perf_counter() - t_infer_start

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels_out)
    y_topk = np.concatenate(all_topk)

    top2_acc = float(np.mean(np.any(y_true[:, None] == y_topk[:, :2], axis=1)))
    top3_acc = float(np.mean(np.any(y_true[:, None] == y_topk[:, :3], axis=1)))

    with h5py.File(h5_path, "r") as f:
        val_runtimes = f["runtimes"][sorted(val_idx)]
    order        = np.argsort(np.argsort(val_idx))
    val_runtimes = val_runtimes[order]

    metrics        = compute_metrics(y_true, y_pred, n_classes)
    noa_5          = near_optimal_accuracy(y_pred, val_runtimes, 0.05)
    noa_10         = near_optimal_accuracy(y_pred, val_runtimes, 0.10)
    noa_20         = near_optimal_accuracy(y_pred, val_runtimes, 0.20)
    mrr_mean, mrr_med = mean_runtime_ratio(y_pred, val_runtimes)
    fail_rate, fail_n = failure_rate(y_pred, val_runtimes)

    with h5py.File(h5_path, "r") as f:
        full_labels = f["labels"][:]
    our_counts = np.bincount(full_labels, minlength=n_classes)

    train_time_s   = ckpt.get("training_time_s", float("nan"))
    n_val          = len(y_true)
    infer_per_ms   = (infer_total_s / n_val) * 1000

    def _fmt_time(s):
        if s != s:   # nan
            return "n/a (old checkpoint)"
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m:02d}m {sec:02d}s  ({s:.0f}s total)" if h else \
               f"{m}m {sec:02d}s  ({s:.0f}s total)"

    print(f"\n── Thesis Experiment Results: {EXPERIMENT} ──────────────────────────")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Checkpoint   : epoch {ckpt['epoch']}")
    print(f"  Val samples  : {n_val}  (of {len(dataset)} total)")
    print(f"  Classes      : {n_classes}")
    print(f"  Model        : {ckpt.get('model_size','small')}  |  "
          f"in_channels={ckpt.get('in_channels',1)}  |  params={n_params:,}")
    print(f"  Training time: {_fmt_time(train_time_s)}")
    print(f"  Inference    : {infer_total_s*1000:.1f}ms total  ({infer_per_ms:.3f}ms/sample on {DEVICE})")
    print()
    print(f"  Accuracy (Acc) : {metrics['Acc']*100:.2f}%   (paper: 78.54%)")
    print(f"  Macro Precision: {metrics['MP']*100:.2f}%   (paper: 63.41%)")
    print(f"  Macro Recall   : {metrics['MR']*100:.2f}%   (paper: 62.81%)")
    print(f"  Macro F1       : {metrics['F1']*100:.2f}%   (paper: 62.53%)")
    print()
    print(f"  Top-2 Accuracy : {top2_acc*100:.2f}%")
    print(f"  Top-3 Accuracy : {top3_acc*100:.2f}%")
    print()
    print(f"  Near-optimal Acc (±5%)  : {noa_5*100:.2f}%")
    print(f"  Near-optimal Acc (±10%) : {noa_10*100:.2f}%")
    print(f"  Near-optimal Acc (±20%) : {noa_20*100:.2f}%")
    print()
    print(f"  Mean runtime ratio      : {mrr_mean:.3f}x  (median: {mrr_med:.3f}x)")
    print(f"  Failure rate            : {fail_rate*100:.2f}%  ({fail_n}/{len(y_true)} predictions chose a non-converging solver)")
    print()
    # Per-class failure rate: among val samples of class i that we predicted as i,
    # how many had NaN runtime for that solver?
    pred_runtimes = val_runtimes[np.arange(len(y_pred)), y_pred]
    print(f"  {'Solver':<20} {'Our N':>7}  {'Paper N':>7}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Fail%':>6}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for i, name in enumerate(SOLVER_NAMES):
        paper_n   = PAPER_COUNTS.get(name, 0)
        pred_mask = y_pred == i
        n_pred    = pred_mask.sum()
        n_fail_i  = int(np.sum(~np.isfinite(pred_runtimes[pred_mask]))) if n_pred > 0 else 0
        fail_pct  = 100.0 * n_fail_i / n_pred if n_pred > 0 else 0.0
        print(f"  {name:<20} {our_counts[i]:>7}  {paper_n:>7}  "
              f"{metrics['per_class_f1'][i]*100:>5.1f}%  "
              f"{metrics['per_class_p'][i]*100:>5.1f}%  "
              f"{metrics['per_class_r'][i]*100:>5.1f}%  "
              f"{fail_pct:>5.1f}%")
    print("─────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
