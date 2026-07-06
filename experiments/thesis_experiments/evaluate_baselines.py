"""
evaluate_baselines.py — Non-learning baselines for solver selection.

Evaluates three baselines on the same stratified 90/10 split used by train.py
(seed=42) so results are directly comparable:

  Majority class  — always predicts the most frequent class in the training set
  1-NN            — nearest neighbour by Euclidean distance on standardised features
  5-NN            — 5-nearest neighbours (majority vote)

Output format matches evaluate.py so results integrate cleanly into
results_summary.txt.

Environment variables:
  DATA_DIR    Directory containing dataset.h5  (default /workspace/data)
  VAL_SPLIT   Validation fraction              (default 0.10)
"""

import os
import sys
import time

import h5py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from model import SOLVER_NAMES, N_SOLVERS

DATA_DIR  = os.getenv("DATA_DIR",  "/workspace/data")
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.10"))
H5_PATH   = os.path.join(DATA_DIR, "dataset.h5")

# ── load data ─────────────────────────────────────────────────────────────────

print(f"[load] Reading {H5_PATH} ...", flush=True)
t0 = time.perf_counter()
with h5py.File(H5_PATH, "r") as f:
    features = f["features"][:]
    labels   = f["labels"][:]
print(f"[load] Done in {time.perf_counter() - t0:.1f}s  "
      f"— {len(labels)} samples, {features.shape[1]} features", flush=True)

# Replace NaN/inf in features (same clamping as train.py)
features = np.nan_to_num(features, nan=0.0, posinf=6e4, neginf=-6e4)
features = np.clip(features, -6e4, 6e4)

# ── stratified split (identical to train.py) ──────────────────────────────────

print("[split] Building stratified train/val split (seed=42) ...", flush=True)
rng = np.random.default_rng(42)
train_idx, val_idx = [], []
for cls in np.unique(labels):
    idx    = np.where(labels == cls)[0]
    n_val  = max(1, int(len(idx) * VAL_SPLIT))
    chosen = rng.choice(idx, size=n_val, replace=False)
    train_idx.extend(np.setdiff1d(idx, chosen).tolist())
    val_idx.extend(chosen.tolist())

train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)

X_train, y_train = features[train_idx], labels[train_idx]
X_val,   y_val   = features[val_idx],   labels[val_idx]

print(f"[split] Train: {len(train_idx)}   Val: {len(val_idx)}", flush=True)

print("[scale] Fitting StandardScaler on training features ...", flush=True)
t0 = time.perf_counter()
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
print(f"[scale] Done in {time.perf_counter() - t0:.2f}s", flush=True)

# ── class distribution ────────────────────────────────────────────────────────

train_counts = np.bincount(y_train, minlength=N_SOLVERS)
val_counts   = np.bincount(y_val,   minlength=N_SOLVERS)

print("\n── Class distribution ──────────────────────────────────────")
print(f"  {'Solver':<30}  {'Train':>6}  {'Val':>5}  {'Train%':>7}")
print(f"  {'-'*30}  {'------':>6}  {'-----':>5}  {'-------':>7}")
for i, name in enumerate(SOLVER_NAMES):
    pct = 100.0 * train_counts[i] / max(len(train_idx), 1)
    print(f"  {name:<30}  {train_counts[i]:>6}  {val_counts[i]:>5}  {pct:>6.1f}%")
print(f"  {'TOTAL':<30}  {len(train_idx):>6}  {len(val_idx):>5}")

# ── helpers ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    mp  = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    mr  = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    return acc, mp, mr, f1


def print_result(name: str, y_true, y_pred, elapsed: float) -> tuple:
    acc, mp, mr, f1 = metrics(y_true, y_pred)
    print(f"  Accuracy (Acc)  : {acc:>6.2f}%")
    print(f"  Macro Precision : {mp:>6.2f}%")
    print(f"  Macro Recall    : {mr:>6.2f}%")
    print(f"  Macro F1        : {f1:>6.2f}%")
    print(f"  Time            : {elapsed:.2f}s")
    return acc, mp, mr, f1


def per_class_breakdown(y_true, y_pred, title: str):
    counts = np.bincount(y_true, minlength=N_SOLVERS)
    print(f"\n  {'Solver':<30}  {'Total':>5}  {'Correct':>7}  {'Acc%':>6}")
    print(f"  {'-'*30}  {'-----':>5}  {'-------':>7}  {'------':>6}")
    for i, name in enumerate(SOLVER_NAMES):
        mask  = y_true == i
        total = mask.sum()
        if total == 0:
            continue
        correct = (y_pred[mask] == i).sum()
        acc_i   = 100.0 * correct / total
        print(f"  {name:<30}  {total:>5}  {correct:>7}  {acc_i:>5.1f}%")


# ── header ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Non-learning Baselines")
print(f"  Dataset : {H5_PATH}")
print(f"  Train   : {len(train_idx)}   Val: {len(val_idx)}")
print("=" * 60)

results = {}

# ── majority class ────────────────────────────────────────────────────────────

t0 = time.perf_counter()
majority_cls  = int(np.bincount(y_train).argmax())
y_majority    = np.full_like(y_val, majority_cls)
elapsed       = time.perf_counter() - t0

print(f"\n── Majority Class  (always predicts: {SOLVER_NAMES[majority_cls]}) ──")
results["majority"] = print_result("majority", y_val, y_majority, elapsed)
per_class_breakdown(y_val, y_majority, "majority")

# ── KNN baselines ─────────────────────────────────────────────────────────────

for k in (1, 5):
    print(f"\n[knn] Fitting {k}-NN on {len(X_train_s)} training samples ...", flush=True)
    t0  = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(X_train_s, y_train)
    t_fit = time.perf_counter() - t0

    print(f"[knn] Predicting on {len(X_val_s)} validation samples ...", flush=True)
    t1     = time.perf_counter()
    y_pred = knn.predict(X_val_s)
    t_pred = time.perf_counter() - t1
    elapsed = time.perf_counter() - t0
    per_query_us = (t_pred / len(X_val_s)) * 1e6

    print(f"[knn] fit={t_fit:.2f}s  predict={t_pred:.2f}s  ({per_query_us:.1f}µs/query on {len(X_train_s)} train vectors)", flush=True)
    print(f"\n── {k}-NN  (Euclidean distance, standardised features) ──")
    print(f"  Train vectors stored : {len(X_train_s)}  (must be scanned per query)")
    print(f"  Fit time (store data): {t_fit*1000:.1f}ms")
    print(f"  Predict time total   : {t_pred*1000:.1f}ms  ({per_query_us:.1f}µs/query)")
    results[f"{k}-nn"] = print_result(f"{k}-nn", y_val, y_pred, elapsed)
    per_class_breakdown(y_val, y_pred, f"{k}-nn")

# ── per-class classification_report for 5-NN ─────────────────────────────────

print("\n── 5-NN full classification report ─────────────────────────")
knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
knn5.fit(X_train_s, y_train)
y_knn5 = knn5.predict(X_val_s)

report = classification_report(
    y_val, y_knn5,
    target_names=SOLVER_NAMES,
    digits=3,
    zero_division=0,
)
print(report)

# ── summary table (matches append_results format in run_experiments.sh) ───────

print("\nexperiment           Acc%    MP%     MR%     F1%")
print("----------------------------------------------------")
for name, (acc, mp, mr, f1) in results.items():
    print(f"{name:<20} {acc:>7.2f} {mp:>7.2f} {mr:>7.2f} {f1:>7.2f}")
print()
