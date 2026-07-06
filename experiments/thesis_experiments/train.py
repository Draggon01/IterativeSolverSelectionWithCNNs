"""
train_solver_selector.py — Train the CNN+MLP solver-selection classifier.

Reads:   $DATA_DIR/dataset.h5          (produced by generate_data.py)
Writes:  $CHECKPOINT_DIR/<EXPERIMENT>/epoch_NNNN.pt
         $LOG_DIR/<EXPERIMENT>/        (TensorBoard event files)

Environment variables (all optional):
  EXPERIMENT       Run name for checkpoint/log subdirectory (default "default")
                   e.g. EXPERIMENT=v2_density128 IMAGE_MODE=density IMAGE_SIZE=128
  DATA_DIR         Path to the HDF5 dataset    (default /workspace/data)
  CHECKPOINT_DIR   Checkpoint root directory   (default /workspace/checkpoints)
  LOG_DIR          TensorBoard root directory  (default /workspace/logs)
  MAX_EPOCHS       Total training epochs       (default 100)
  BATCH_SIZE       Mini-batch size             (default 256)
  LEARNING_RATE    Initial AdamW learning rate (default 3e-4)
  CHECKPOINT_EVERY Save a checkpoint every N epochs (default 5)
  KEEP_LAST_N      Number of checkpoints to retain (default 3)
  VAL_SPLIT        Fraction of data used for validation (default 0.10)
  DEVICE           "cpu", "cuda", or "auto"   (default auto)
"""

import os
import glob
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import h5py

from model import SolverSelectorNet, N_FEATURES, N_SOLVERS, SOLVERS, IMAGE_SIZE, MODEL_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
EXPERIMENT       = os.getenv("EXPERIMENT",      "default")
_DATA_DIR        = os.getenv("DATA_DIR",        "/workspace/data")
_CHECKPOINT_ROOT = os.getenv("CHECKPOINT_DIR",  "/workspace/checkpoints")
_LOG_ROOT        = os.getenv("LOG_DIR",          "/workspace/logs")
DATA_DIR         = _DATA_DIR
CHECKPOINT_DIR   = os.path.join(_CHECKPOINT_ROOT, EXPERIMENT)
LOG_DIR          = os.path.join(_LOG_ROOT,         EXPERIMENT)
MAX_EPOCHS       = int(os.getenv("MAX_EPOCHS",   "100"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",   "256"))
LR               = float(os.getenv("LEARNING_RATE", "3e-4"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "5"))
KEEP_LAST_N      = int(os.getenv("KEEP_LAST_N",  "3"))
VAL_SPLIT        = float(os.getenv("VAL_SPLIT",  "0.10"))

_dev = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)
MIXED_PRECISION      = os.getenv("MIXED_PRECISION", "1") == "1" and DEVICE.type == "cuda"
NO_CNN               = os.getenv("NO_CNN", "0") == "1"
CONVERGENCE_PENALTY  = float(os.getenv("CONVERGENCE_PENALTY", "0.0"))
# MODEL_SIZE and IMAGE_MODE2 are read from model.py module-level env vars


# ── dataset ───────────────────────────────────────────────────────────────────

class SolverDataset(Dataset):
    """
    Lazy-loading HDF5 dataset compatible with multi-worker DataLoader.

    Each item is (image, features, label, converge_mask) where:
        image        — (C, H, W) float32 sparsity-pattern tensor
        features     — (N_FEATURES,) float32 matrix statistics tensor
        label        — int class index into SOLVERS
        converge_mask — (N_SOLVERS,) float32: 1.0 where solver converged, 0.0 where it diverged
    """

    def __init__(self, h5_path: str):
        self.path = h5_path
        # Read metadata without keeping the file open (fork-safe)
        with h5py.File(h5_path, "r") as f:
            self.length     = len(f["labels"])
            self.n_features = f["features"].shape[1]
            self.n_solvers  = len(f.attrs.get("solvers", [None] * N_SOLVERS))
            self.image_mode = str(f.attrs.get("image_mode", "binary"))
            self.has_images  = "images"  in f
            self.has_images2 = "images2" in f
            self.n_channels  = 2 if (self.has_images and self.has_images2) else 1
        self._file = None   # opened lazily inside each worker

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        if self.has_images:
            img = torch.nan_to_num(
                torch.from_numpy(self._file["images"][idx][None]),
                nan=0.0, posinf=1.0, neginf=0.0,
            )
            if self.has_images2:
                img2 = torch.nan_to_num(
                    torch.from_numpy(self._file["images2"][idx][None]),
                    nan=0.0, posinf=1.0, neginf=0.0,
                )
                img = torch.cat([img, img2], dim=0)  # (2, H, W)
        else:
            img = torch.zeros(self.n_channels, IMAGE_SIZE, IMAGE_SIZE)
        feat = torch.nan_to_num(
            torch.from_numpy(self._file["features"][idx]),
            nan=0.0, posinf=6e4, neginf=-6e4,
        ).clamp_(-6e4, 6e4)  # fp16 max ~65504; clamp finite outliers too
        lbl          = int(self._file["labels"][idx])
        runtimes     = self._file["runtimes"][idx]          # (N_SOLVERS,) float32, NaN = diverged
        converge_mask = torch.from_numpy(
            np.isfinite(runtimes).astype(np.float32)        # 1.0 = converged, 0.0 = diverged
        )
        return img, feat, lbl, converge_mask


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: SolverSelectorNet,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_acc: float,
    image_mode: str = "binary",
    scaler: "GradScaler | None" = None,
    training_time_s: float = 0.0,
) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:04d}.pt")
    ckpt = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "val_acc":          val_acc,
        "n_features":       model.stats[0].in_features,
        "n_classes":        model.head[-1].out_features,
        "image_mode":       image_mode,
        "no_cnn":           model.no_cnn,
        "model_size":       model.model_size,
        "in_channels":      model.in_channels,
        "training_time_s":  training_time_s,
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)
    log.info("Checkpoint saved: %s  (val_acc=%.4f)", path, val_acc)

    # Remove oldest checkpoints beyond the retention limit
    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    for old in existing[:-KEEP_LAST_N]:
        os.remove(old)
        log.debug("Removed old checkpoint: %s", old)


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model:               SolverSelectorNet,
    loader:              DataLoader,
    criterion:           nn.Module,
    optimizer:           optim.Optimizer,
    device:              torch.device,
    train:               bool,
    scaler:              "GradScaler | None" = None,
    convergence_penalty: float = 0.0,
) -> tuple[float, float]:
    model.train(train)
    total_loss = correct = total = 0
    n_batches = len(loader)
    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"

    with torch.set_grad_enabled(train):
        for batch_idx, (img, feat, lbl, converge_mask) in enumerate(loader):
            img, feat, lbl = img.to(device), feat.to(device), lbl.to(device)
            converge_mask  = converge_mask.to(device)      # (batch, N_SOLVERS)

            with autocast(device_type=amp_device, enabled=(scaler is not None)):
                logits  = torch.clamp(model(img, feat), -50, 50)
                ce_loss = criterion(logits, lbl)

                if convergence_penalty > 0.0 and train:
                    # Penalise probability mass assigned to non-converging solvers.
                    # diverge_mask = 1 where solver did NOT converge for this sample.
                    probs        = torch.softmax(logits, dim=1)
                    diverge_mask = 1.0 - converge_mask
                    penalty      = (probs * diverge_mask).sum(dim=1).mean()
                    loss         = ce_loss + convergence_penalty * penalty
                else:
                    loss = ce_loss

            if torch.isnan(loss):
                log.warning("NaN loss on batch %d — skipping", batch_idx)
                if train:
                    optimizer.zero_grad()
                continue

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            bs          = len(lbl)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == lbl).sum().item()
            total      += bs

            if train and (batch_idx + 1) % max(1, n_batches // 5) == 0:
                log.info("  batch %d/%d  loss=%.4f  acc=%.4f",
                         batch_idx + 1, n_batches,
                         total_loss / total, correct / total)

    if total == 0:
        return float("nan"), 0.0
    return total_loss / total, correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    dataset = SolverDataset(h5_path)

    # Stratified split: each class contributes VAL_SPLIT fraction to val
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "r") as f:
        all_labels = f["labels"][:]
    train_idx, val_idx = [], []
    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        n_cls_val = max(1, int(len(idx) * VAL_SPLIT))
        chosen_val = rng.choice(idx, size=n_cls_val, replace=False)
        train_idx.extend(np.setdiff1d(idx, chosen_val).tolist())
        val_idx.extend(chosen_val.tolist())
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              drop_last=(len(train_ds) > BATCH_SIZE))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Inverse-frequency class weights so rare classes (250 samples) get equal
    # gradient signal as dominant ones (600 samples) despite the imbalance.
    train_labels = all_labels[train_idx]
    counts = np.bincount(train_labels, minlength=dataset.n_solvers).astype(np.float32)
    class_weights = torch.tensor(
        len(train_idx) / (dataset.n_solvers * np.maximum(counts, 1)),
        dtype=torch.float32,
    ).to(DEVICE)

    model     = SolverSelectorNet(
                    dataset.n_features, dataset.n_solvers,
                    no_cnn=NO_CNN, model_size=MODEL_SIZE,
                    in_channels=dataset.n_channels,
                ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler    = GradScaler() if MIXED_PRECISION else None
    writer    = SummaryWriter(log_dir=LOG_DIR)
    log.info("Class weights (min=%.3f max=%.3f): %s",
             class_weights.min().item(), class_weights.max().item(),
             {SOLVERS[i]: f"{class_weights[i].item():.2f}" for i in class_weights.argsort()[:3].tolist()
              + class_weights.argsort()[-3:].tolist()})
    log.info("Mixed precision (AMP): %s", "enabled" if scaler else "disabled")
    log.info("CNN branch: %s", "disabled (features only)" if NO_CNN else "enabled")
    log.info("Model size: %s  |  in_channels: %d", MODEL_SIZE, dataset.n_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{n_params:,}")
    log.info("Convergence penalty λ: %.2f", CONVERGENCE_PENALTY)

    # Resume from the latest checkpoint in this experiment's directory if one exists
    start_epoch      = 1
    prev_train_time  = 0.0
    existing         = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    ckpt_path        = existing[-1] if existing else None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        ckpt_size     = ckpt.get("model_size",  "small")
        ckpt_channels = ckpt.get("in_channels", 1)
        if ckpt_size != MODEL_SIZE or ckpt_channels != dataset.n_channels:
            log.warning(
                "Checkpoint mismatch (size=%r channels=%d) vs current (size=%r channels=%d)"
                " — starting from scratch.",
                ckpt_size, ckpt_channels, MODEL_SIZE, dataset.n_channels,
            )
            ckpt_path = None
        else:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and "scaler_state" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch     = ckpt["epoch"] + 1
            prev_train_time = ckpt.get("training_time_s", 0.0)
            log.info("Resumed from %s  (epoch %d)", ckpt_path, ckpt["epoch"])

    log.info(
        "Experiment=%s | device=%s | train=%d val=%d | epochs=%d batch=%d lr=%.1e",
        EXPERIMENT, DEVICE, len(train_idx), len(val_idx), MAX_EPOCHS, BATCH_SIZE, LR,
    )

    t_run_start = time.perf_counter()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True,  scaler=scaler, convergence_penalty=CONVERGENCE_PENALTY)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False, scaler=None,   convergence_penalty=0.0)
        scheduler.step()

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc",  {"train": tr_acc,  "val": va_acc},  epoch)
        writer.add_scalar("lr",    scheduler.get_last_lr()[0],          epoch)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f",
            epoch, MAX_EPOCHS, tr_loss, tr_acc, va_loss, va_acc,
        )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == MAX_EPOCHS:
            elapsed = prev_train_time + (time.perf_counter() - t_run_start)
            save_checkpoint(model, optimizer, scheduler, epoch, va_acc,
                            dataset.image_mode, scaler, training_time_s=elapsed)

    total_time = prev_train_time + (time.perf_counter() - t_run_start)
    h, m = divmod(int(total_time), 3600)
    m, s = divmod(m, 60)
    log.info("Training complete.  Total wall time: %dh %02dm %02ds (%.0fs)",
             h, m, s, total_time)
    writer.close()


if __name__ == "__main__":
    main()
