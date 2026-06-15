"""
render.py — Re-render images from a base HDF5 dataset at a new mode/size.

Reads a source dataset that was generated with STORE_MATRIX=1, reconstructs
each sparse matrix from the stored CSR data, and writes a new HDF5 with
re-rendered images.  All other fields (features, labels, runtimes, source,
top3_labels, mat_*) are copied as-is.

This avoids re-running PETSc solvers when exploring different image modes
or sizes — just re-render from the stored matrix data.

Environment variables:
  SRC_DATA_DIR   Source directory containing dataset.h5 with matrix data
                 (default /workspace/data/base)
  DATA_DIR       Output directory for the new dataset.h5
                 (default /workspace/data)
  IMAGE_MODE     binary | density | log_density | magnitude  (default binary)
  IMAGE_SIZE     Output image resolution in pixels           (default 64)
  BATCH_SIZE     Matrices rendered per log message           (default 500)
"""

import logging
import os
import shutil

import h5py
import numpy as np
import scipy.sparse as sp

from model import sparsity_image, IMAGE_MODE, IMAGE_SIZE, SOLVER_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_DATA_DIR = os.getenv("SRC_DATA_DIR", "/workspace/data/base")
DATA_DIR     = os.getenv("DATA_DIR",     "/workspace/data")
MODE         = IMAGE_MODE   # from model.py env var
SIZE         = IMAGE_SIZE   # from model.py env var
LOG_EVERY    = int(os.getenv("BATCH_SIZE", "500"))


def main() -> None:
    src_path = os.path.join(SRC_DATA_DIR, "dataset.h5")
    dst_path = os.path.join(DATA_DIR, "dataset.h5")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    with h5py.File(src_path, "r") as src:
        if not src.attrs.get("has_matrix_data", False):
            raise RuntimeError(
                f"{src_path} was not generated with STORE_MATRIX=1. "
                "Re-run datagen with STORE_MATRIX=1 to enable re-rendering."
            )
        n = len(src["labels"])
        log.info("Source: %s  (%d samples)", src_path, n)
        log.info("Rendering  mode=%s  size=%d → %s", MODE, SIZE, dst_path)

        os.makedirs(DATA_DIR, exist_ok=True)
        tmp_path = dst_path + ".tmp"

        with h5py.File(tmp_path, "w") as dst:
            # Copy non-image datasets verbatim
            copy_keys = [k for k in src.keys() if k != "images"]
            for key in copy_keys:
                src.copy(key, dst)

            # Create new images dataset at the target size
            ds_img = dst.create_dataset(
                "images",
                shape=(n, SIZE, SIZE),
                dtype="f4",
                chunks=(min(64, n), SIZE, SIZE),
            )

            # Copy attributes, update image metadata
            for attr_key, attr_val in src.attrs.items():
                dst.attrs[attr_key] = attr_val
            dst.attrs["image_mode"] = MODE
            dst.attrs["image_size"] = SIZE

            # Re-render each matrix
            for i in range(n):
                data    = src["mat_data"][i].astype(np.float64)
                indices = src["mat_indices"][i].astype(np.int32)
                indptr  = src["mat_indptr"][i].astype(np.int32)
                shape   = tuple(src["mat_shape"][i])
                A = sp.csr_matrix((data, indices, indptr), shape=shape)
                ds_img[i] = sparsity_image(A, size=SIZE, mode=MODE)

                if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                    log.info("  Rendered %d / %d", i + 1, n)

        # Atomic replace
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.rename(tmp_path, dst_path)

    log.info("Done — wrote %s", dst_path)


if __name__ == "__main__":
    main()
