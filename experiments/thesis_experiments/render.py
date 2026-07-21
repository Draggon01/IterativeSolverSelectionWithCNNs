"""
render.py — Re-render images from a base HDF5 dataset.

Two modes selected by the MODES environment variable:

  Single/dual-channel (MODES not set):
    Renders IMAGE_MODE into 'images', and optionally IMAGE_MODE2 into 'images2'.
    Output: DATA_DIR/dataset.h5  (mirrors base with re-rendered images)

  Multimode (MODES="mode1 mode2 ..."):
    Renders each mode into a separate 'images_<mode>' dataset in one shared file.
    Output: DATA_DIR/dataset.h5  (used by ensemble_evaluate.py)
    Skips modes already present — safe to re-run incrementally.

Matrix reconstruction strategy (both modes):
  - synthetic/* rows: read stored CSR arrays (requires STORE_MATRIX=1 during datagen)
  - suitesparse/*   : re-download via ssgetpy (cached in CACHE_DIR)
  - manual/*        : reload from the original .mtx path on disk (single mode only)

Environment variables:
  SRC_DATA_DIR   Base dataset.h5 with stored matrices  (default /workspace/data/base)
  DATA_DIR       Output directory                      (default /workspace/data)
  CACHE_DIR      SuiteSparse .mtx cache                (default SRC_DATA_DIR/suitesparse_cache)
  IMAGE_SIZE     Pixel resolution                      (default 64)
  IMAGE_MODE     Single/dual mode: primary channel     (default binary)
  IMAGE_MODE2    Single/dual mode: second channel, or empty for single-channel
  MODES          Multimode: space-separated list of modes to render
"""

import glob
import logging
import os

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

from model import sparsity_image, IMAGE_MODE, IMAGE_MODE2, IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR", "/workspace/data/base")
DATA_DIR     = os.getenv("DATA_DIR",     "/workspace/data")
CACHE_DIR    = os.getenv("CACHE_DIR",    os.path.join(SRC_DATA_DIR, "suitesparse_cache"))
SIZE         = IMAGE_SIZE
_modes_env   = os.getenv("MODES", "")
MODES        = _modes_env.split() if _modes_env.strip() else None  # None = single/dual mode

_default_modes = "magnitude signed_magnitude rcm_signed_magnitude symmetry rcm_magnitude density"

LOG_EVERY = 200


# ── Shared matrix reconstruction helpers ─────────────────────────────────────

def _has_stored_csr(src: h5py.File, i: int) -> bool:
    return (
        "mat_data" in src
        and i < len(src["mat_data"])
        and len(src["mat_data"][i]) > 0
        and len(src["mat_indptr"][i]) > 0
    )


def _load_from_stored(src: h5py.File, i: int) -> sp.csr_matrix:
    data    = src["mat_data"][i].astype(np.float64)
    indices = src["mat_indices"][i].astype(np.int32)
    indptr  = src["mat_indptr"][i].astype(np.int32)
    shape   = tuple(src["mat_shape"][i])
    return sp.csr_matrix((data, indices, indptr), shape=shape)


def _load_from_suitesparse(source: str) -> sp.csr_matrix:
    import ssgetpy
    _, group, name = source.split("/", 2)
    pattern = os.path.join(CACHE_DIR, name, "*.mtx")
    files   = glob.glob(pattern)
    if not files:
        results = ssgetpy.search(name=name, group=group)
        if not results:
            raise RuntimeError(f"Matrix not found in SuiteSparse index: {source}")
        results[0].download(destpath=CACHE_DIR, format="MM", extract=True)
        files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No .mtx file found after download for {source}")
    exact  = [f for f in files if os.path.splitext(os.path.basename(f))[0] == name]
    chosen = exact[0] if exact else sorted(files)[0]
    return sp.csr_matrix(scipy.io.mmread(chosen))


def _load_from_manual(source: str) -> sp.csr_matrix:
    rel_path = source.removeprefix("manual/") + ".mtx"
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"Manual matrix file not found: {rel_path}")
    return sp.csr_matrix(scipy.io.mmread(rel_path))


def reconstruct_matrix(src: h5py.File, i: int, allow_none: bool = False) -> sp.csr_matrix | None:
    """Return matrix i from stored data or by re-downloading."""
    raw_source = src["source"][i]
    source = raw_source.decode() if isinstance(raw_source, bytes) else raw_source

    if _has_stored_csr(src, i):
        return _load_from_stored(src, i)

    if source.startswith("suitesparse/"):
        try:
            return _load_from_suitesparse(source)
        except Exception as e:
            if allow_none:
                log.warning("Row %d (%s): failed to load — %s. Using zero image.", i, source, e)
                return None
            raise

    if source.startswith("manual/"):
        return _load_from_manual(source)

    if allow_none:
        log.warning("Row %d (source=%r): no stored CSR and no supported fallback. Using zero image.", i, source)
        return None

    raise RuntimeError(
        f"Row {i} (source='{source}') has no stored CSR data and no supported "
        "re-download strategy. Re-run datagen with STORE_MATRIX=1 for synthetic rows."
    )


# ── Single / dual-channel rendering ──────────────────────────────────────────

def render_single(src_path: str, dst_path: str) -> None:
    mode  = IMAGE_MODE
    mode2 = IMAGE_MODE2

    with h5py.File(src_path, "r") as src:
        n = len(src["labels"])
        mode_str = f"{mode} + {mode2}" if mode2 else mode
        log.info("Source : %s  (%d samples)", src_path, n)
        log.info("Mode   : %s  size=%d → %s", mode_str, SIZE, dst_path)

        if "mat_data" not in src:
            log.warning(
                "No mat_data in source. Only suitesparse/* and manual/* rows can be re-rendered."
            )

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = dst_path + ".tmp"

        with h5py.File(tmp_path, "w") as dst:
            skip = {"images", "images2", "mat_data", "mat_indices", "mat_indptr", "mat_shape"}
            for key in (k for k in src.keys() if k not in skip):
                src.copy(key, dst)

            ds_img = dst.create_dataset("images", shape=(n, SIZE, SIZE), dtype="f4",
                                        chunks=(min(64, n), SIZE, SIZE))
            ds_img2 = None
            if mode2:
                ds_img2 = dst.create_dataset("images2", shape=(n, SIZE, SIZE), dtype="f4",
                                             chunks=(min(64, n), SIZE, SIZE))

            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["image_mode"] = mode
            dst.attrs["image_size"] = SIZE
            if mode2:
                dst.attrs["image_mode2"] = mode2

            for i in range(n):
                A = reconstruct_matrix(src, i)
                ds_img[i] = sparsity_image(A, size=SIZE, mode=mode)
                if ds_img2 is not None:
                    ds_img2[i] = sparsity_image(A, size=SIZE, mode=mode2)
                if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                    log.info("  Rendered %d / %d", i + 1, n)

        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)


# ── Multimode rendering ───────────────────────────────────────────────────────

def render_multimode(src_path: str, dst_path: str, modes: list[str]) -> None:
    dst_exists = os.path.exists(dst_path)

    if dst_exists:
        with h5py.File(dst_path, "r") as f:
            modes_to_render = [m for m in modes if f"images_{m}" not in f]
    else:
        modes_to_render = list(modes)

    if not modes_to_render:
        log.info("All requested modes already present in %s — nothing to do.", dst_path)
        return

    log.info("Modes to render : %s", modes_to_render)
    log.info("Modes skipped   : %s", [m for m in modes if m not in modes_to_render])

    with h5py.File(src_path, "r") as src:
        n = len(src["labels"])
        log.info("Source : %s  (%d samples)", src_path, n)
        log.info("Size   : %d px  output: %s  (append=%s)", SIZE, dst_path, dst_exists)

        os.makedirs(DATA_DIR, exist_ok=True)
        with h5py.File(dst_path, "a" if dst_exists else "w") as dst:
            if not dst_exists:
                for key in ("features", "labels", "runtimes", "source"):
                    if key in src:
                        src.copy(key, dst)
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["image_size"] = SIZE

            mode_datasets = {
                m: dst.create_dataset(f"images_{m}", shape=(n, SIZE, SIZE), dtype="f4",
                                      chunks=(min(64, n), SIZE, SIZE))
                for m in modes_to_render
            }
            for m in modes_to_render:
                log.info("  Created dataset images_%s", m)

            skipped = 0
            for i in range(n):
                A = reconstruct_matrix(src, i, allow_none=True)
                if A is None:
                    skipped += 1
                    for m in modes_to_render:
                        mode_datasets[m][i] = np.zeros((SIZE, SIZE), dtype=np.float32)
                    continue
                for m in modes_to_render:
                    mode_datasets[m][i] = sparsity_image(A, size=SIZE, mode=m)
                if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                    log.info("  Rendered %d / %d  (skipped: %d)", i + 1, n, skipped)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    src_path = os.path.join(SRC_DATA_DIR, "dataset.h5")
    dst_path = os.path.join(DATA_DIR, "dataset.h5")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    if MODES is not None:
        modes = MODES if MODES else _default_modes.split()
        render_multimode(src_path, dst_path, modes)
    else:
        render_single(src_path, dst_path)

    log.info("Done — wrote %s", dst_path)


if __name__ == "__main__":
    main()
