"""
ensemble_evaluate.py — Evaluate an ensemble of trained models by averaging softmax outputs.

Reads images from a pre-rendered multimode HDF5 (data/multimode/dataset.h5). If the file
is missing or lacks any required mode, render_multimode.py is run automatically first.

Environment variables:
  EXPERIMENTS      Space-separated experiment names to ensemble
  MULTIMODE_DIR    Directory containing the multimode dataset.h5
                   (default: /workspace/data/multimode)
  SRC_DATA_DIR     Base dataset for render_multimode if needed
                   (default: /workspace/data/base)
  IMAGE_SIZE       Image size used when auto-rendering  (default: 64)
  CHECKPOINT_DIR   Checkpoint root  (default /workspace/checkpoints)
  VAL_SPLIT        Validation fraction  (default 0.10)
  DEVICE           cpu | cuda | auto    (default auto)
  RESULTS_FILE     If set, append all printed output (with header) to this file
"""

import glob
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import torch

from model import SolverSelectorNet, SOLVER_NAMES, N_SOLVERS
from evaluate import (
    compute_metrics,
    near_optimal_accuracy,
    mean_runtime_ratio,
    failure_rate,
    PAPER_COUNTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_CKPT_ROOT    = os.getenv("CHECKPOINT_DIR",  "/workspace/checkpoints")
MULTIMODE_DIR = os.getenv("MULTIMODE_DIR",   "/workspace/data/multimode")
SRC_DATA_DIR  = os.getenv("SRC_DATA_DIR",    "/workspace/data/base")
IMAGE_SIZE    = int(os.getenv("IMAGE_SIZE",  "64"))
VAL_SPLIT     = float(os.getenv("VAL_SPLIT", "0.10"))
RESULTS_FILE  = os.getenv("RESULTS_FILE",    "")


class _Tee:
    """Write to both the original stdout and a file simultaneously."""
    def __init__(self, original, filepath: str):
        self._orig = original
        self._f    = open(filepath, "a", buffering=1)

    def write(self, data: str) -> int:
        self._orig.write(data)
        return self._f.write(data)

    def flush(self) -> None:
        self._orig.flush()
        self._f.flush()

    def close(self) -> None:
        self._f.close()

_dev   = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)

_default_experiments = (
    "magnitude__signed_magnitude_64 "
    "magnitude__rcm_signed_magnitude_64 "
    "magnitude__symmetry_64 "
    "rcm_magnitude__signed_magnitude_64"
)
EXPERIMENTS = os.getenv("EXPERIMENTS", _default_experiments).split()


def load_model(experiment: str) -> tuple[SolverSelectorNet, dict]:
    ckpt_dir = os.path.join(_CKPT_ROOT, experiment)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    model = SolverSelectorNet(
        ckpt["n_features"], ckpt["n_classes"],
        no_cnn=ckpt.get("no_cnn", False),
        model_size=ckpt.get("model_size", "small"),
        in_channels=ckpt.get("in_channels", 1),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def get_val_indices(all_labels: np.ndarray) -> list[int]:
    rng = np.random.default_rng(42)
    val_idx = []
    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        n_val = max(1, int(len(idx) * VAL_SPLIT))
        val_idx.extend(rng.choice(idx, size=n_val, replace=False).tolist())
    return val_idx


def infer_image_modes(ckpt: dict, experiment: str) -> tuple[str, str]:
    """
    Extract image_mode and image_mode2 from a checkpoint, falling back to
    parsing the experiment name.  Checkpoints store image_mode but not
    image_mode2 (added later), so always parse mode2 from the name.
    Naming convention: <mode1>__<mode2>_<size> (dual) or <mode>_<size> (single).
    """
    # Strip trailing _<size> (e.g. _64, _128, _256)
    name = experiment.rsplit("_", 1)[0]

    if "__" in name:
        parsed_mode1, parsed_mode2 = name.split("__", 1)
    else:
        parsed_mode1, parsed_mode2 = name, ""

    mode1 = ckpt.get("image_mode", "") or parsed_mode1
    mode2 = ckpt.get("image_mode2", "") or parsed_mode2

    return mode1, mode2


def run_inference(
    model: SolverSelectorNet,
    ckpt: dict,
    experiment: str,
    f: h5py.File,
    val_idx: list[int],
) -> np.ndarray:
    """
    Read pre-rendered images from the multimode HDF5 and run inference.
    Returns softmax probabilities (N_val, N_classes).
    """
    mode1, mode2 = infer_image_modes(ckpt, experiment)
    key1 = f"images_{mode1}"
    key2 = f"images_{mode2}" if mode2 else None

    if key1 not in f:
        raise KeyError(
            f"Mode '{mode1}' not found in multimode dataset (key '{key1}'). "
            f"Re-run render_multimode with MODES including '{mode1}'."
        )
    if key2 and key2 not in f:
        raise KeyError(
            f"Mode '{mode2}' not found in multimode dataset (key '{key2}'). "
            f"Re-run render_multimode with MODES including '{mode2}'."
        )

    all_probs = []
    BATCH = 256

    for start in range(0, len(val_idx), BATCH):
        batch_orig = val_idx[start:start + BATCH]         # original order
        sort_order   = np.argsort(batch_orig)             # positions that sort batch
        unsort_order = np.argsort(sort_order)             # inverse: restores original order
        batch_sorted = [batch_orig[i] for i in sort_order]

        # h5py requires sorted indices for fancy indexing
        img1 = f[key1][batch_sorted]                      # (B, H, W)
        img1 = torch.from_numpy(img1[:, None, :, :])      # (B, 1, H, W)
        img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=0.0)

        if key2:
            img2 = f[key2][batch_sorted]
            img2 = torch.from_numpy(img2[:, None, :, :])
            img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=0.0)
            imgs = torch.cat([img1, img2], dim=1)          # (B, 2, H, W)
        else:
            imgs = img1                                    # (B, 1, H, W)

        feat = f["features"][batch_sorted]
        feat = np.nan_to_num(feat, nan=0.0, posinf=6e4, neginf=-6e4)
        feat = np.clip(feat, -6e4, 6e4)
        feats = torch.from_numpy(feat)

        with torch.no_grad():
            logits = model(imgs.to(DEVICE), feats.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs[unsort_order])  # restore original batch order

    return np.concatenate(all_probs, axis=0)   # (N_val, N_classes) in val_idx order


def ensure_multimode(needed_modes: list[str]) -> None:
    """Run render_multimode.py to add any missing modes to the multimode HDF5."""
    multimode_h5 = os.path.join(MULTIMODE_DIR, "dataset.h5")

    if os.path.exists(multimode_h5):
        with h5py.File(multimode_h5, "r") as f:
            stored_size = f.attrs.get("image_size", IMAGE_SIZE)
            missing = [m for m in needed_modes if f"images_{m}" not in f]
        if stored_size != IMAGE_SIZE:
            log.info("Multimode HDF5 is at %dpx but IMAGE_SIZE=%d — deleting and re-rendering.", stored_size, IMAGE_SIZE)
            os.remove(multimode_h5)
            missing = needed_modes
        elif not missing:
            log.info("Multimode HDF5 already has all needed modes at %dpx — skipping render.", IMAGE_SIZE)
            return
        else:
            log.info("Multimode HDF5 missing modes %s — rendering ...", missing)
    else:
        log.info("Multimode HDF5 not found — rendering now ...")
        missing = needed_modes

    env = os.environ.copy()
    env["SRC_DATA_DIR"] = SRC_DATA_DIR
    env["DATA_DIR"]     = MULTIMODE_DIR
    env["IMAGE_SIZE"]   = str(IMAGE_SIZE)
    env["MODES"]        = " ".join(missing)   # only render what's actually missing

    script = os.path.join(os.path.dirname(__file__), "render_multimode.py")
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        raise RuntimeError("render_multimode.py failed — check output above.")


def main() -> None:
    tee = None
    if RESULTS_FILE:
        os.makedirs(os.path.dirname(os.path.abspath(RESULTS_FILE)), exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n============================================================\n")
            f.write(f"  Experiment : ensemble ({len(EXPERIMENTS)} models)\n")
            f.write(f"  Members    : {', '.join(EXPERIMENTS)}\n")
            f.write(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"============================================================\n")
        tee = _Tee(sys.stdout, RESULTS_FILE)
        sys.stdout = tee

    try:
        _main()
    finally:
        if tee is not None:
            sys.stdout = tee._orig
            tee.close()


def _main() -> None:
    # Determine which modes are needed across all experiments
    needed_modes: list[str] = []
    for exp in EXPERIMENTS:
        # Load checkpoint just for the mode fields; full model loaded later
        ckpt_dir = os.path.join(_CKPT_ROOT, exp)
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
            m1, m2 = infer_image_modes(ckpt, exp)
        else:
            m1, m2 = infer_image_modes({}, exp)
        if m1 and m1 not in needed_modes:
            needed_modes.append(m1)
        if m2 and m2 not in needed_modes:
            needed_modes.append(m2)

    ensure_multimode(needed_modes)

    multimode_h5 = os.path.join(MULTIMODE_DIR, "dataset.h5")

    print(f"\n── Ensemble Evaluation ───────────────────────────────────────────")
    print(f"  Models        : {len(EXPERIMENTS)}")
    for i, exp in enumerate(EXPERIMENTS):
        print(f"  [{i+1}] {exp}")
    print(f"  Multimode HDF5: {multimode_h5}")
    print(f"  Device        : {DEVICE}")
    print()

    with h5py.File(multimode_h5, "r") as f:
        all_labels = f["labels"][:]

    val_idx = get_val_indices(all_labels)
    y_true  = all_labels[val_idx]

    with h5py.File(multimode_h5, "r") as f:
        order        = np.argsort(np.argsort(val_idx))
        val_runtimes = f["runtimes"][sorted(val_idx)][order]

    all_model_probs = []
    n_classes = None

    t0 = time.perf_counter()
    with h5py.File(multimode_h5, "r") as f:
        for exp in EXPERIMENTS:
            model, ckpt = load_model(exp)
            if n_classes is None:
                n_classes = ckpt["n_classes"]

            mode1, mode2 = infer_image_modes(ckpt, exp)
            log.info("Running %s  mode=%s+%s  model=%s  ch=%d  epoch=%d",
                     exp, mode1, mode2 or "-", ckpt.get("model_size","small"),
                     ckpt.get("in_channels",1), ckpt["epoch"])

            probs = run_inference(model, ckpt, exp, f, val_idx)
            ind_acc = 100.0 * np.mean(np.argmax(probs, axis=1) == y_true)
            ind_f1  = compute_metrics(y_true, np.argmax(probs, axis=1), n_classes)["F1"]
            log.info("  → individual Acc=%.2f%%  F1=%.2f%%", ind_acc, ind_f1 * 100)
            all_model_probs.append(probs)

    infer_total_s = time.perf_counter() - t0

    ensemble_probs = np.mean(all_model_probs, axis=0)
    y_pred = np.argmax(ensemble_probs, axis=1)
    y_topk = np.argsort(-ensemble_probs, axis=1)[:, :3]

    top2_acc = float(np.mean(np.any(y_true[:, None] == y_topk[:, :2], axis=1)))
    top3_acc = float(np.mean(np.any(y_true[:, None] == y_topk[:, :3], axis=1)))

    metrics           = compute_metrics(y_true, y_pred, n_classes)
    noa_5             = near_optimal_accuracy(y_pred, val_runtimes, 0.05)
    noa_10            = near_optimal_accuracy(y_pred, val_runtimes, 0.10)
    noa_20            = near_optimal_accuracy(y_pred, val_runtimes, 0.20)
    mrr_mean, mrr_med = mean_runtime_ratio(y_pred, val_runtimes)
    fail_rate, fail_n = failure_rate(y_pred, val_runtimes)

    n_val        = len(y_true)
    infer_per_ms = (infer_total_s / n_val) * 1000

    print(f"── Ensemble Results ──────────────────────────────────────────────")
    print(f"  Members      : {', '.join(EXPERIMENTS)}")
    print(f"  Val samples  : {n_val}")
    print(f"  Classes      : {n_classes}")
    print(f"  Inference    : {infer_total_s*1000:.1f}ms total  ({infer_per_ms:.3f}ms/sample)")
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
    print(f"  Failure rate            : {fail_rate*100:.2f}%  ({fail_n}/{n_val})")
    print()
    print(f"  Individual model breakdown:")
    for exp, probs in zip(EXPERIMENTS, all_model_probs):
        ind_acc = 100.0 * np.mean(np.argmax(probs, axis=1) == y_true)
        ind_f1  = compute_metrics(y_true, np.argmax(probs, axis=1), n_classes)["F1"]
        print(f"    {exp:<45} Acc={ind_acc:.2f}%  F1={ind_f1*100:.2f}%")
    print()

    our_counts    = np.bincount(all_labels, minlength=n_classes)
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
