"""
render.py — Re-render images from a base HDF5 dataset at a new mode/size.

For each row the matrix is reconstructed by one of two strategies:
  - synthetic/* rows: read stored CSR arrays (requires STORE_MATRIX=1 during datagen)
  - suitesparse/*  : re-download via ssgetpy (cached in CACHE_DIR; no stored data needed)
  - manual/*       : reload from the original .mtx path on disk

All other fields (features, labels, runtimes, source, top3_labels, mat_*)
are copied verbatim to the output file.

Environment variables:
  SRC_DATA_DIR   Source directory containing dataset.h5
                 (default /workspace/data/base)
  DATA_DIR       Output directory for the re-rendered dataset.h5
                 (default /workspace/data)
  CACHE_DIR      Cache directory for SuiteSparse downloads
                 (default SRC_DATA_DIR/suitesparse_cache)
  IMAGE_MODE     binary | density | log_density | magnitude  (default binary)
  IMAGE_SIZE     Output image resolution in pixels           (default 64)
  BATCH_SIZE     Matrices rendered per log message           (default 500)
"""

import glob
import logging
import os
import shutil

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

from model import sparsity_image, IMAGE_MODE, IMAGE_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR", "/workspace/data/base")
DATA_DIR     = os.getenv("DATA_DIR",     "/workspace/data")
CACHE_DIR    = os.getenv("CACHE_DIR",    os.path.join(SRC_DATA_DIR, "suitesparse_cache"))
MODE         = IMAGE_MODE
SIZE         = IMAGE_SIZE
LOG_EVERY    = int(os.getenv("BATCH_SIZE", "500"))


def _has_stored_csr(src: h5py.File, i: int) -> bool:
    return (
        "mat_data" in src
        and i < len(src["mat_data"])
        and len(src["mat_data"][i]) > 0
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
    return sp.csr_matrix(scipy.io.mmread(files[0]))


def _load_from_manual(source: str) -> sp.csr_matrix:
    # source = "manual/relative/path/to/matrix"
    rel_path = source.removeprefix("manual/") + ".mtx"
    if not os.path.exists(rel_path):
        raise FileNotFoundError(f"Manual matrix file not found: {rel_path}")
    return sp.csr_matrix(scipy.io.mmread(rel_path))


def reconstruct_matrix(src: h5py.File, i: int) -> sp.csr_matrix:
    """Return matrix i using stored CSR data or re-downloading as appropriate."""
    raw_source = src["source"][i]
    source = raw_source.decode() if isinstance(raw_source, bytes) else raw_source

    if _has_stored_csr(src, i):
        return _load_from_stored(src, i)

    if source.startswith("suitesparse/"):
        log.debug("Re-downloading %s", source)
        return _load_from_suitesparse(source)

    if source.startswith("manual/"):
        return _load_from_manual(source)

    raise RuntimeError(
        f"Row {i} (source='{source}') has no stored CSR data and no supported "
        "re-download strategy. Re-run datagen with STORE_MATRIX=1 for synthetic rows."
    )


def main() -> None:
    src_path = os.path.join(SRC_DATA_DIR, "dataset.h5")
    dst_path = os.path.join(DATA_DIR, "dataset.h5")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    with h5py.File(src_path, "r") as src:
        n = len(src["labels"])
        log.info("Source: %s  (%d samples)", src_path, n)
        log.info("Rendering  mode=%s  size=%d → %s", MODE, SIZE, dst_path)

        # Warn about synthetic rows that lack stored CSR data
        has_any_csr = "mat_data" in src
        if not has_any_csr:
            log.warning(
                "No mat_data in source file. Only suitesparse/* and manual/* "
                "rows can be re-rendered. Synthetic rows will fail."
            )

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = dst_path + ".tmp"

        with h5py.File(tmp_path, "w") as dst:
            copy_keys = [k for k in src.keys() if k != "images"]
            for key in copy_keys:
                src.copy(key, dst)

            ds_img = dst.create_dataset(
                "images",
                shape=(n, SIZE, SIZE),
                dtype="f4",
                chunks=(min(64, n), SIZE, SIZE),
            )

            for attr_key, attr_val in src.attrs.items():
                dst.attrs[attr_key] = attr_val
            dst.attrs["image_mode"] = MODE
            dst.attrs["image_size"] = SIZE

            for i in range(n):
                A = reconstruct_matrix(src, i)
                ds_img[i] = sparsity_image(A, size=SIZE, mode=MODE)

                if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                    log.info("  Rendered %d / %d", i + 1, n)

        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)

    log.info("Done — wrote %s", dst_path)


if __name__ == "__main__":
    main()
