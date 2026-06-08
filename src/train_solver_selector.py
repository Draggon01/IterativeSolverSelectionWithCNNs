"""
train_solver_selector.py — Train the CNN+MLP solver-selection classifier.

Reads:   $DATA_DIR/dataset.h5          (produced by generate_data.py)
Writes:  $CHECKPOINT_DIR/epoch_NNNN.pt (keeps the last KEEP_LAST_N)
         $LOG_DIR/                     (TensorBoard event files)

Environment variables (all optional):
  DATA_DIR         Path to the HDF5 dataset    (default /workspace/data)
  CHECKPOINT_DIR   Checkpoint output directory (default /workspace/checkpoints)
  LOG_DIR          TensorBoard log directory   (default /workspace/logs)
  MAX_EPOCHS       Total training epochs       (default 100)
  BATCH_SIZE       Mini-batch size             (default 256)
  LEARNING_RATE    Initial AdamW learning rate (default 3e-4)
  CHECKPOINT_EVERY Save a checkpoint every N epochs (default 5)
  KEEP_LAST_N      Number of checkpoints to retain (default 3)
  VAL_SPLIT        Fraction of data used for validation (default 0.15)
  DEVICE           "cpu", "cuda", or "auto"   (default auto)
"""

import os
import glob
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import h5py

from model import SolverSelectorNet, N_FEATURES, N_SOLVERS, SOLVERS, latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR         = os.getenv("DATA_DIR",        "/workspace/data")
CHECKPOINT_DIR   = os.getenv("CHECKPOINT_DIR",  "/workspace/checkpoints")
LOG_DIR          = os.getenv("LOG_DIR",          "/workspace/logs")
MAX_EPOCHS       = int(os.getenv("MAX_EPOCHS",   "100"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",   "256"))
LR               = float(os.getenv("LEARNING_RATE", "3e-4"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "5"))
KEEP_LAST_N      = int(os.getenv("KEEP_LAST_N",  "3"))
VAL_SPLIT        = float(os.getenv("VAL_SPLIT",  "0.15"))

_dev = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)


# ── dataset ───────────────────────────────────────────────────────────────────

class SolverDataset(Dataset):
    """
    Lazy-loading HDF5 dataset compatible with multi-worker DataLoader.

    Each item is (image, features, label) where:
        image    — (1, H, W) float32 sparsity-pattern tensor
        features — (N_FEATURES,) float32 matrix statistics tensor
        label    — int class index into SOLVERS
    """

    def __init__(self, h5_path: str):
        self.path = h5_path
        # Read metadata without keeping the file open (fork-safe)
        with h5py.File(h5_path, "r") as f:
            self.length     = len(f["labels"])
            self.n_features = f["features"].shape[1]
            self.n_solvers  = len(f.attrs.get("solvers", [None] * N_SOLVERS))
            self.image_mode = str(f.attrs.get("image_mode", "binary"))
        self._file = None   # opened lazily inside each worker

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        img  = torch.from_numpy(self._file["images"][idx][None])   # (1, H, W)
        feat = torch.from_numpy(self._file["features"][idx])       # (F,)
        lbl  = int(self._file["labels"][idx])
        return img, feat, lbl


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: SolverSelectorNet,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    image_mode: str = "binary",
) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc":         val_acc,
            "n_features":      model.stats[0].in_features,
            "n_classes":       model.head[-1].out_features,
            "image_mode":      image_mode,
        },
        path,
    )
    log.info("Checkpoint saved: %s  (val_acc=%.4f)", path, val_acc)

    # Remove oldest checkpoints beyond the retention limit
    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    for old in existing[:-KEEP_LAST_N]:
        os.remove(old)
        log.debug("Removed old checkpoint: %s", old)


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model:     SolverSelectorNet,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    train:     bool,
) -> tuple[float, float]:
    """Run one epoch; return (mean_loss, accuracy)."""
    model.train(train)
    total_loss = correct = total = 0

    with torch.set_grad_enabled(train):
        for img, feat, lbl in loader:
            img, feat, lbl = img.to(device), feat.to(device), lbl.to(device)
            logits = model(img, feat)
            loss   = criterion(logits, lbl)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs          = len(lbl)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == lbl).sum().item()
            total      += bs

    return total_loss / total, correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset   = SolverDataset(os.path.join(DATA_DIR, "dataset.h5"))
    n_val     = max(1, int(len(dataset) * VAL_SPLIT))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = SolverSelectorNet(dataset.n_features, dataset.n_solvers).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    writer    = SummaryWriter(log_dir=LOG_DIR)

    # Resume from the latest checkpoint if one exists
    start_epoch = 1
    ckpt_path   = latest_checkpoint()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        log.info("Resumed from %s  (epoch %d)", ckpt_path, ckpt["epoch"])

    log.info(
        "Training on %s | train=%d  val=%d | epochs=%d  batch=%d  lr=%.1e",
        DEVICE, n_train, n_val, MAX_EPOCHS, BATCH_SIZE, LR,
    )

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)
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
            save_checkpoint(model, optimizer, epoch, va_acc, dataset.image_mode)

    writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
