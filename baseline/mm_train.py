"""
mm_train.py — Train the MM-AutoSolver baseline model.

Hyperparameters match the paper (Table 4):
  Epochs 256, Batch 512, Optimizer Adam, LR 1e-3, Loss CrossEntropy.

Reads:   $DATA_DIR/dataset.h5         (produced by mm_generate.py)
Writes:  $CHECKPOINT_DIR/epoch_*.pt
         $LOG_DIR/                    (TensorBoard event files)

Environment variables (all optional):
  DATA_DIR        HDF5 dataset directory  (default /workspace/data)
  CHECKPOINT_DIR  Checkpoint directory    (default /workspace/checkpoints)
  LOG_DIR         TensorBoard log dir     (default /workspace/logs)
  MAX_EPOCHS      Training epochs         (default 256, matches paper)
  BATCH_SIZE      Mini-batch size         (default 512, matches paper)
  LEARNING_RATE   Adam learning rate      (default 1e-3, matches paper)
  VAL_SPLIT       Validation fraction     (default 0.10, matches paper's 1/10 split)
  CHECKPOINT_EVERY  Save every N epochs   (default 10)
  KEEP_LAST_N     Checkpoints to retain   (default 3)
  DEVICE          cpu | cuda | auto       (default auto)
"""

import glob
import logging
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from mm_model import MMAutoSolverNet, MM_N_FEATURES, MM_IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR         = os.getenv("DATA_DIR",        "./data")
CHECKPOINT_DIR   = os.getenv("CHECKPOINT_DIR",  "./checkpoints")
LOG_DIR          = os.getenv("LOG_DIR",          "./logs")
MAX_EPOCHS       = int(os.getenv("MAX_EPOCHS",   "256"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE",   "512"))
LR               = float(os.getenv("LEARNING_RATE", "1e-3"))
VAL_SPLIT        = float(os.getenv("VAL_SPLIT",  "0.10"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))
KEEP_LAST_N      = int(os.getenv("KEEP_LAST_N",  "3"))

_dev   = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)


# ── dataset ───────────────────────────────────────────────────────────────────

class MMDataset(Dataset):
    """Lazy-loading HDF5 dataset for the MM-AutoSolver baseline."""

    def __init__(self, h5_path: str):
        self.path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length    = len(f["labels"])
            self.n_classes = len(f.attrs.get("solvers", []))
        self._file = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        img  = torch.nan_to_num(
            torch.from_numpy(self._file["images"][idx][None]),     # (1, H, W)
            nan=0.0, posinf=1.0, neginf=0.0,
        )
        feat = torch.nan_to_num(
            torch.from_numpy(self._file["features"][idx]),         # (17,)
            nan=0.0, posinf=1e6, neginf=-1e6,
        )
        lbl  = int(self._file["labels"][idx])
        return img, feat, lbl


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:04d}.pt")
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc":         val_acc,
        "n_features":      MM_N_FEATURES,
        "n_classes":       model.pred_head.out_features,
    }, path)
    log.info("Checkpoint saved: %s  (val_acc=%.4f)", path, val_acc)

    existing = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    for old in existing[:-KEEP_LAST_N]:
        os.remove(old)


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = correct = total = 0
    n_batches = len(loader)

    with torch.set_grad_enabled(train):
        for batch_idx, (img, feat, lbl) in enumerate(loader):
            img, feat, lbl = img.to(device), feat.to(device), lbl.to(device)
            logits = model(img, feat)
            logits = torch.clamp(logits, -50, 50)
            loss   = criterion(logits, lbl)

            if torch.isnan(loss):
                log.warning("  NaN loss on batch %d — skipping", batch_idx)
                if train:
                    optimizer.zero_grad()
                continue

            if train:
                optimizer.zero_grad()
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

    dataset  = MMDataset(os.path.join(DATA_DIR, "dataset.h5"))

    # Stratified split: each class contributes VAL_SPLIT fraction to val
    rng = np.random.default_rng(42)
    with h5py.File(os.path.join(DATA_DIR, "dataset.h5"), "r") as f:
        all_labels = f["labels"][:]
    train_idx, val_idx = [], []
    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        n_cls_val = max(1, int(len(idx) * VAL_SPLIT))
        chosen_val = rng.choice(idx, size=n_cls_val, replace=False)
        chosen_train = np.setdiff1d(idx, chosen_val)
        val_idx.extend(chosen_val.tolist())
        train_idx.extend(chosen_train.tolist())
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = MMAutoSolverNet(MM_N_FEATURES, dataset.n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    writer    = SummaryWriter(log_dir=LOG_DIR)

    log.info(
        "MM-AutoSolver baseline | device=%s | train=%d val=%d | "
        "epochs=%d batch=%d lr=%.0e",
        DEVICE, len(train_idx), len(val_idx), MAX_EPOCHS, BATCH_SIZE, LR,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, False)

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc",  {"train": tr_acc,  "val": va_acc},  epoch)

        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f",
            epoch, MAX_EPOCHS, tr_loss, tr_acc, va_loss, va_acc,
        )

        if epoch % CHECKPOINT_EVERY == 0 or epoch == MAX_EPOCHS:
            save_checkpoint(model, optimizer, epoch, va_acc)

    writer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
