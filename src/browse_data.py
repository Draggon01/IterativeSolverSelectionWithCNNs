"""
browse_data.py — Interactive matrix browser for the training dataset.

Displays one matrix at a time with:
  Left panel   — sparsity-pattern image, border colour = best solver
  Centre panel — per-solver prediction probabilities (requires checkpoint)
  Right panel  — toggleable view (cycle with the [View] button):
                   Features  : normalised bar chart of matrix statistics
                   Runtimes  : actual PETSc solver timings in milliseconds
                               (only available if dataset was generated with
                               the updated generate_data.py that saves runtimes)
                   Info      : formatted text summary of all matrix properties

Controls:
  ◀ Prev / Next ▶   step through samples (wraps)
  ⟳ Random           jump to a random sample
  Go-to box          type a dataset index and press Enter
  [View] button      cycle right-panel between Features / Runtimes / Info
  Solver filter      show only samples where a specific solver is the best

Run from src/:
  python browse_data.py
  DATA_DIR=./data CHECKPOINT_DIR=./checkpoints python browse_data.py

Requires a display.  On a remote server use X11 forwarding (ssh -X) or
run the non-interactive visualize.py instead.
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox, RadioButtons
import h5py
import torch

from model import SOLVERS, N_SOLVERS, N_FEATURES, load_checkpoint, SolverSelectorNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR       = os.getenv("DATA_DIR",       "/workspace/data")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/workspace/checkpoints")

_dev   = os.getenv("DEVICE", "auto")
DEVICE = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
)

PALETTE      = plt.cm.tab10(np.linspace(0, 0.9, N_SOLVERS))
SOLVER_COLOR = {s: PALETTE[i] for i, s in enumerate(SOLVERS)}

FEATURE_NAMES = [
    "log(n)", "log(nnz)", "density", "symmetry",
    "diag dom.", "Frob/n", "trace/n", "max/mean",
]

RIGHT_MODES   = ["Features", "Runtimes", "Info"]


# ── browser ───────────────────────────────────────────────────────────────────

class MatrixBrowser:
    """
    Interactive matplotlib window for browsing individual matrices in dataset.h5.
    """

    def __init__(
        self,
        images:   np.ndarray,            # (N, H, W)        float32
        features: np.ndarray,            # (N, N_FEATURES)  float32
        labels:   np.ndarray,            # (N,)             int32
        runtimes: "np.ndarray | None",   # (N, N_SOLVERS)   float32, NaN = no data
        top3:     "np.ndarray | None",   # (N, 3)           int8,    -1 = no rank
        sources:  "np.ndarray | None",   # (N,)             str
        model:    "SolverSelectorNet | None" = None,
    ) -> None:
        self.images   = images
        self.features = features
        self.labels   = labels
        self.runtimes = runtimes
        self.top3     = top3
        self.sources  = sources
        self.model    = model
        self.n_total  = len(labels)

        self._filter     = -1              # -1 = all; 0-5 = solver index
        self._indices    = np.arange(self.n_total)
        self._pos        = 0
        self._right_mode = 0               # 0=Features 1=Runtimes 2=Info

        self._build_figure()
        self._draw()
        plt.show()

    # ── index helpers ─────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        self._indices = (
            np.arange(self.n_total)
            if self._filter == -1
            else np.where(self.labels == self._filter)[0]
        )
        self._pos = 0

    @property
    def _current(self) -> int:
        if len(self._indices) == 0:
            return 0
        return int(self._indices[self._pos % len(self._indices)])

    def _step(self, delta: int) -> None:
        n = len(self._indices)
        if n:
            self._pos = (self._pos + delta) % n
        self._draw()

    # ── figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.patch.set_facecolor("#f5f5f5")

        # Three equal plot columns in the upper portion
        plot_gs = gridspec.GridSpec(
            1, 3,
            left=0.04, right=0.99,
            top=0.91, bottom=0.36,
            wspace=0.32,
        )
        self.ax_img  = self.fig.add_subplot(plot_gs[0])
        self.ax_prob = self.fig.add_subplot(plot_gs[1])
        self.ax_right = self.fig.add_subplot(plot_gs[2])

        # Info strip
        self.ax_info = self.fig.add_axes([0.04, 0.27, 0.92, 0.07])
        self.ax_info.axis("off")
        self._info_txt = self.ax_info.text(
            0.5, 0.5, "",
            transform=self.ax_info.transAxes,
            ha="center", va="center",
            fontsize=10, family="monospace",
        )

        # Navigation buttons
        bh, by = 0.07, 0.17
        self.btn_prev  = Button(self.fig.add_axes([0.04, by, 0.08, bh]), "◀  Prev")
        self.btn_next  = Button(self.fig.add_axes([0.13, by, 0.08, bh]), "Next  ▶")
        self.btn_rand  = Button(self.fig.add_axes([0.22, by, 0.09, bh]), "⟳ Random")
        self.tbox      = TextBox(self.fig.add_axes([0.35, by, 0.07, bh]), "Go to:", initial="0")
        self.btn_mode  = Button(
            self.fig.add_axes([0.44, by, 0.09, bh]),
            f"View:\n{RIGHT_MODES[self._right_mode]}",
        )

        self.btn_prev.on_clicked(lambda _: self._step(-1))
        self.btn_next.on_clicked(lambda _: self._step(+1))
        self.btn_rand.on_clicked(self._on_random)
        self.tbox.on_submit(self._on_goto)
        self.btn_mode.on_clicked(self._on_mode_toggle)

        # Solver filter radio buttons
        radio_opts = ["all"] + SOLVERS
        self.radio  = RadioButtons(
            self.fig.add_axes([0.55, 0.02, 0.44, 0.22]),
            radio_opts, active=0,
        )
        try:
            for label, circle in zip(radio_opts, self.radio.circles):
                color = "#aaaaaa" if label == "all" else SOLVER_COLOR[label]
                circle.set_facecolor(color)
                circle.set_radius(0.05)
        except AttributeError:
            pass
        self.radio.on_clicked(self._on_filter)
        self.fig.text(0.55, 0.255, "Filter by best solver:", fontsize=9, style="italic")

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_random(self, _event) -> None:
        n = len(self._indices)
        if n:
            self._pos = int(np.random.randint(0, n))
        self._draw()

    def _on_goto(self, text: str) -> None:
        try:
            target = int(text)
            if 0 <= target < self.n_total:
                where = np.where(self._indices == target)[0]
                self._pos = int(where[0]) if len(where) else 0
        except ValueError:
            pass
        self._draw()

    def _on_filter(self, label: str) -> None:
        self._filter = -1 if label == "all" else SOLVERS.index(label)
        self._rebuild_index()
        self._draw()

    def _on_mode_toggle(self, _event) -> None:
        self._right_mode = (self._right_mode + 1) % len(RIGHT_MODES)
        self.btn_mode.label.set_text(f"View:\n{RIGHT_MODES[self._right_mode]}")
        self._draw()

    # ── inference ─────────────────────────────────────────────────────────────

    def _predict(self, idx: int) -> "np.ndarray | None":
        if self.model is None:
            return None
        img_t  = torch.from_numpy(self.images[idx][None, None]).to(DEVICE)
        feat_t = torch.from_numpy(self.features[idx][None]).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(img_t, feat_t), dim=1)
        return probs.cpu().numpy()[0]

    # ── main draw dispatcher ──────────────────────────────────────────────────

    def _draw(self) -> None:
        if len(self._indices) == 0:
            for ax in (self.ax_img, self.ax_prob, self.ax_right):
                ax.cla()
                ax.text(0.5, 0.5, "No samples match filter",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=11)
            self.fig.canvas.draw_idle()
            return

        idx    = self._current
        img    = self.images[idx]
        feat   = self.features[idx]
        label  = int(self.labels[idx])
        probs  = self._predict(idx)
        color  = SOLVER_COLOR[SOLVERS[label]]
        rt     = self.runtimes[idx] if self.runtimes is not None else None
        top3   = self.top3[idx]     if self.top3     is not None else None
        source = (self.sources[idx] if self.sources  is not None else "")
        if isinstance(source, bytes):
            source = source.decode()

        self._draw_image(img, label, color)
        self._draw_probs(probs, label)

        if self._right_mode == 0:
            self._draw_features(feat)
        elif self._right_mode == 1:
            self._draw_runtimes(rt, label)
        else:
            self._draw_matrix_info(feat, label, probs, top3, source)

        self._draw_info_strip(feat, label, probs, idx, rt, top3)
        self.tbox.set_val(str(idx))
        self.fig.canvas.draw_idle()

    # ── panel renderers ───────────────────────────────────────────────────────

    def _draw_image(self, img: np.ndarray, label: int, color) -> None:
        ax = self.ax_img
        ax.cla()
        ax.imshow(img, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(
            f"Sparsity Pattern  —  best: {SOLVERS[label].upper()}",
            fontsize=10, color=color, fontweight="bold",
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    def _draw_probs(self, probs: "np.ndarray | None", true_label: int) -> None:
        ax = self.ax_prob
        ax.cla()

        if probs is None:
            ax.text(0.5, 0.5,
                    "No checkpoint found.\nRun train_solver_selector.py first.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=10)
            ax.set_title("Model Prediction", fontsize=10)
            ax.axis("off")
            return

        pred    = int(probs.argmax())
        correct = pred == true_label
        colors  = [SOLVER_COLOR[s] for s in SOLVERS]
        bars    = ax.barh(SOLVERS, probs, color=colors, alpha=0.85)

        for bar, p, i in zip(bars, probs, range(N_SOLVERS)):
            ax.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{p:.2f}", va="center", fontsize=8.5)
            if i == true_label:
                bar.set_edgecolor("black"); bar.set_linewidth(2)

        status      = "✓ correct" if correct else "✗ wrong"
        title_color = "#2ca02c" if correct else "#d62728"
        ax.set_title(f"Prediction: {SOLVERS[pred].upper()}  {status}",
                     fontsize=10, color=title_color, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability")
        ax.set_yticks(range(N_SOLVERS)); ax.set_yticklabels(SOLVERS)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.text(0.99, -0.09, "■ = actual best",
                transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    def _draw_features(self, feat: np.ndarray) -> None:
        ax = self.ax_right
        ax.cla()
        norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-9)
        bars = ax.barh(FEATURE_NAMES, norm, color="steelblue", alpha=0.75)
        for bar, raw, nv in zip(bars, feat, norm):
            ax.text(min(nv + 0.02, 0.97), bar.get_y() + bar.get_height() / 2,
                    f"{raw:.3g}", va="center", fontsize=8.5)
        ax.set_xlim(0, 1.25)
        ax.set_title("Matrix Statistics  (normalised)", fontsize=10)
        ax.set_xlabel("Relative magnitude")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    def _draw_runtimes(
        self,
        rt:    "np.ndarray | None",   # (N_SOLVERS,) in seconds, NaN = no data
        label: int,
    ) -> None:
        ax = self.ax_right
        ax.cla()

        if rt is None or np.all(np.isnan(rt)):
            ax.text(
                0.5, 0.5,
                "No runtime data.\nRegenerate dataset with\nupdated generate_data.py.",
                ha="center", va="center", transform=ax.transAxes,
                color="gray", fontsize=10,
            )
            ax.set_title("Solver Runtimes", fontsize=10)
            ax.axis("off")
            return

        ms     = rt * 1000.0                   # convert to milliseconds
        colors = [SOLVER_COLOR[s] for s in SOLVERS]
        bars   = ax.barh(
            SOLVERS, ms,
            color=colors, alpha=0.85,
        )

        # Grey out non-converged bars
        for i, (bar, val) in enumerate(zip(bars, ms)):
            if np.isnan(val):
                bar.set_width(0)
                ax.text(1, bar.get_y() + bar.get_height() / 2,
                        "n/c", va="center", fontsize=8, color="gray")
            else:
                ax.text(val + max(ms[~np.isnan(ms)]) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f} ms", va="center", fontsize=8.5)
            if i == label:
                bar.set_edgecolor("black"); bar.set_linewidth(2)

        fastest = SOLVERS[label]
        ax.set_title(f"Solver Runtimes  —  fastest: {fastest.upper()}",
                     fontsize=10, color=SOLVER_COLOR[fastest], fontweight="bold")
        ax.set_xlabel("Wall-clock time (ms)")
        ax.set_yticks(range(N_SOLVERS)); ax.set_yticklabels(SOLVERS)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.text(0.99, -0.09, "■ = fastest (label)",
                transform=ax.transAxes, ha="right", fontsize=8, color="gray")

    def _draw_matrix_info(
        self,
        feat:   np.ndarray,
        label:  int,
        probs:  "np.ndarray | None",
        top3:   "np.ndarray | None",
        source: str,
    ) -> None:
        ax = self.ax_right
        ax.cla()
        ax.axis("off")

        n   = int(round(np.expm1(feat[0])))
        nnz = int(round(np.expm1(feat[1])))

        sym_score = feat[3]
        sym_str   = "symmetric  (score≈0)" if sym_score < 0.01 \
                    else f"asymmetric (score={sym_score:.3f})"

        dom     = feat[4]
        dom_str = "diagonally dominant" if dom >= 1.0 \
                  else f"not dominant  (ratio={dom:.2f})"

        pred_str = (
            f"{SOLVERS[int(probs.argmax())].upper()}  (conf {probs.max():.0%})"
            if probs is not None else "—  (no model)"
        )

        # Top-3 ranking string
        if top3 is not None:
            ranked = [SOLVERS[int(i)].upper() for i in top3 if int(i) >= 0]
            top3_str = "  >  ".join(ranked) if ranked else "—"
        else:
            top3_str = "—  (regenerate dataset)"

        lines = [
            ("Source",          source or "—"),
            ("Dimension",       f"{n:,} × {n:,}"),
            ("Non-zeros",       f"{nnz:,}"),
            ("Density",         f"{feat[2]:.6f}  ({feat[2]*100:.3f}%)"),
            ("Symmetry",        sym_str),
            ("Diag. dominance", dom_str),
            ("‖A‖_F / n",       f"{feat[5]:.4f}"),
            ("tr(A) / n",       f"{feat[6]:.4f}"),
            ("max / mean |a|",  f"{feat[7]:.2f}×"),
            ("",                ""),
            ("Top-3 solvers",   top3_str),
            ("Prediction",      pred_str),
        ]

        y = 0.97
        ax.text(0.5, y + 0.03, "Matrix Properties",
                transform=ax.transAxes, ha="center", fontsize=11,
                fontweight="bold")
        for key, val in lines:
            if key == "":
                y -= 0.03; continue
            ax.text(0.03, y, key + ":", transform=ax.transAxes,
                    fontsize=8.5, color="#555555", va="top")
            ax.text(0.45, y, val, transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top", family="monospace",
                    wrap=True)
            y -= 0.085
            ax.axhline(y=y + 0.02, xmin=0.01, xmax=0.99,
                       color="#dddddd", linewidth=0.5, transform=ax.transAxes)

    def _draw_info_strip(
        self,
        feat:  np.ndarray,
        label: int,
        probs: "np.ndarray | None",
        idx:   int,
        rt:    "np.ndarray | None",
        top3:  "np.ndarray | None",
    ) -> None:
        n    = int(round(np.expm1(feat[0])))
        nnz  = int(round(np.expm1(feat[1])))
        sym  = "sym" if feat[3] < 0.05 else f"asym={feat[3]:.2f}"
        pred = (f"→ pred: {SOLVERS[int(probs.argmax())].upper()}"
                if probs is not None else "")
        rt_str = ""
        if rt is not None and not np.all(np.isnan(rt)):
            rt_str = f"  fastest={np.nanmin(rt)*1000:.1f} ms"
        top3_str = ""
        if top3 is not None:
            ranked = " > ".join(SOLVERS[int(i)].upper() for i in top3 if int(i) >= 0)
            top3_str = f"  top3: {ranked}"
        self._info_txt.set_text(
            f"#{idx}  |  {n:,}×{n:,}  nnz={nnz:,}  density={feat[2]:.5f}  "
            f"{sym}{rt_str}{top3_str}  |  "
            f"actual: {SOLVERS[label].upper()}  {pred}  |  "
            f"{self._pos + 1}/{len(self._indices)} shown  ({self.n_total} total)"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    if not os.path.exists(h5_path):
        log.error("Dataset not found at %s — run generate_data.py first.", h5_path)
        return

    with h5py.File(h5_path, "r") as f:
        images   = f["images"][:]
        features = f["features"][:]
        labels   = f["labels"][:]
        def _load(key):
            return f[key][:] if key in f and len(f[key]) > 0 else None

        runtimes = _load("runtimes")
        top3     = _load("top3_labels")
        sources  = _load("source")

    for name, arr in [("runtimes", runtimes), ("top3_labels", top3), ("source", sources)]:
        if arr is None:
            log.info("No '%s' dataset — regenerate with updated generate_data.py.", name)
    log.info("Loaded %d samples from %s", len(labels), h5_path)

    model = None
    try:
        model, ckpt = load_checkpoint(DEVICE)
        model.eval()
        log.info("Checkpoint loaded (epoch=%d  val_acc=%.4f)",
                 ckpt.get("epoch", -1), ckpt.get("val_acc", float("nan")))
    except FileNotFoundError:
        log.info("No checkpoint found in %s — predictions disabled.", CHECKPOINT_DIR)

    MatrixBrowser(images, features, labels, runtimes, top3, sources, model)


if __name__ == "__main__":
    main()
