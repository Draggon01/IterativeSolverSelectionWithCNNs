"""
render_multimode.py — Render multiple image modes into a single HDF5 file.

Creates data/multimode/dataset.h5 containing:
  images_<mode>   — (N, SIZE, SIZE) float32 for each requested mode
  features        — copied from base
  labels          — copied from base
  runtimes        — copied from base
  source          — copied from base

Matrix reconstruction mirrors render.py: stored CSR data for synthetic rows,
SuiteSparse cache re-download for suitesparse/* rows.

This file is used by ensemble_evaluate.py so it only needs one dataset on disk.

Environment variables:
  SRC_DATA_DIR   Base dataset.h5 with stored matrices  (default /workspace/data/base)
  DATA_DIR       Output directory                      (default /workspace/data/multimode)
  CACHE_DIR      SuiteSparse .mtx cache                (default SRC_DATA_DIR/suitesparse_cache)
  IMAGE_SIZE     Pixel size for all rendered images    (default 64)
  MODES          Space-separated list of modes to render
                 (default: magnitude signed_magnitude rcm_signed_magnitude
                           symmetry rcm_magnitude density)
"""

import glob
import logging
import os

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

from model import sparsity_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR", "/workspace/data/base")
DATA_DIR     = os.getenv("DATA_DIR",     "/workspace/data/multimode")
CACHE_DIR    = os.getenv("CACHE_DIR",    os.path.join(SRC_DATA_DIR, "suitesparse_cache"))
IMAGE_SIZE   = int(os.getenv("IMAGE_SIZE", "64"))
_default_modes = "magnitude signed_magnitude rcm_signed_magnitude symmetry rcm_magnitude density"
MODES        = os.getenv("MODES", _default_modes).split()

LOG_EVERY = 200


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


def reconstruct_matrix(src: h5py.File, i: int) -> sp.csr_matrix | None:
    """
    Return the sparse matrix for row i, mirroring render.py's strategy:
      - synthetic rows: read stored CSR arrays
      - suitesparse/*: re-load from CACHE_DIR (download if missing)
      - anything else with no stored data: return None (written as zero image)
    """
    if _has_stored_csr(src, i):
        return _load_from_stored(src, i)

    raw_source = src["source"][i]
    source = raw_source.decode() if isinstance(raw_source, bytes) else raw_source

    if source.startswith("suitesparse/"):
        try:
            return _load_from_suitesparse(source)
        except Exception as e:
            log.warning("Row %d (%s): failed to load — %s. Using zero image.", i, source, e)
            return None

    log.warning("Row %d (source=%r): no stored CSR and no supported fallback. Using zero image.", i, source)
    return None


def main() -> None:
    src_path = os.path.join(SRC_DATA_DIR, "dataset.h5")
    dst_path = os.path.join(DATA_DIR, "dataset.h5")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Base dataset not found: {src_path}")

    os.makedirs(DATA_DIR, exist_ok=True)

    dst_exists = os.path.exists(dst_path)

    # Determine which modes actually need rendering (skip already-present ones)
    if dst_exists:
        with h5py.File(dst_path, "r") as f:
            modes_to_render = [m for m in MODES if f"images_{m}" not in f]
    else:
        modes_to_render = list(MODES)

    if not modes_to_render:
        log.info("All requested modes already present in %s — nothing to do.", dst_path)
        return

    log.info("Modes to render : %s", modes_to_render)
    log.info("Modes skipped   : %s", [m for m in MODES if m not in modes_to_render])

    with h5py.File(src_path, "r") as src:
        n = len(src["labels"])
        log.info("Source: %s  (%d samples)", src_path, n)
        log.info("Size  : %d px", IMAGE_SIZE)
        log.info("Output: %s  (append=%s)", dst_path, dst_exists)

        dst_mode = "a" if dst_exists else "w"
        with h5py.File(dst_path, dst_mode) as dst:
            if not dst_exists:
                for key in ("features", "labels", "runtimes", "source"):
                    if key in src:
                        src.copy(key, dst)
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["image_size"] = IMAGE_SIZE

            mode_datasets = {}
            for mode in modes_to_render:
                mode_datasets[mode] = dst.create_dataset(
                    f"images_{mode}",
                    shape=(n, IMAGE_SIZE, IMAGE_SIZE),
                    dtype="f4",
                    chunks=(min(64, n), IMAGE_SIZE, IMAGE_SIZE),
                )
                log.info("  Created dataset images_%s", mode)

            skipped = 0
            for i in range(n):
                A = reconstruct_matrix(src, i)
                if A is None:
                    skipped += 1
                    for mode in modes_to_render:
                        mode_datasets[mode][i] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                    continue
                for mode in modes_to_render:
                    mode_datasets[mode][i] = sparsity_image(A, size=IMAGE_SIZE, mode=mode)

                if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                    log.info("  Rendered %d / %d  (skipped: %d)", i + 1, n, skipped)

    log.info("Done — wrote %s", dst_path)


if __name__ == "__main__":
    main()
