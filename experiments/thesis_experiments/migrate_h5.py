"""
migrate_h5.py — Migrate a dataset.h5 to support extensible solver columns.

Recreates the file with runtimes maxshape=(None, None) so new solver pairs
can be appended later without re-benchmarking existing matrices.

All data is copied unchanged. Only the HDF5 chunking metadata for the
runtimes dataset is updated.

Usage:
    python migrate_h5.py                        # migrates DATA_DIR/dataset.h5
    python migrate_h5.py /path/to/dataset.h5    # explicit path
    python migrate_h5.py /path/to/dir           # uses dir/dataset.h5

Environment:
    DATA_DIR   directory containing dataset.h5  (default /workspace/data)
    CHUNK      rows per copy batch              (default 500)
"""

import logging
import os
import sys

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "/workspace/data")
CHUNK    = int(os.getenv("CHUNK", "500"))


def _resolve_path(arg: str | None) -> str:
    if arg is None:
        return os.path.join(DATA_DIR, "dataset.h5")
    if arg.endswith(".h5"):
        return arg
    return os.path.join(arg, "dataset.h5")


def _copy_dataset(key: str, src: h5py.File, dst: h5py.File) -> None:
    src_ds     = src[key]
    src_shape  = src_ds.shape
    src_dtype  = src_ds.dtype
    item_shape = src_shape[1:]

    if key == "runtimes":
        maxshape = (None, None)
        chunks   = (min(CHUNK, 256), src_shape[1]) if len(src_shape) > 1 else (min(CHUNK, 256),)
    elif not item_shape:
        maxshape = (None,)
        chunks   = (min(CHUNK, 256),)
    else:
        maxshape = (None, *item_shape)
        chunks   = (min(CHUNK, 64), *item_shape)

    vlen = (
        h5py.check_vlen_dtype(src_dtype) is not None
        or h5py.check_string_dtype(src_dtype) is not None
    )

    if vlen:
        dst_ds = dst.create_dataset(
            key, shape=src_shape, maxshape=(None,), dtype=src_dtype, chunks=(256,),
        )
    else:
        dst_ds = dst.create_dataset(
            key, shape=src_shape, maxshape=maxshape, dtype=src_dtype, chunks=chunks,
        )

    n = src_shape[0]
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        dst_ds[start:end] = src_ds[start:end]

    old_max = src_ds.maxshape
    new_max = dst_ds.maxshape
    if key == "runtimes":
        log.info("  %-16s  %d rows  maxshape %s → %s", key, n, old_max, new_max)
    else:
        log.info("  %-16s  %d rows", key, n)


def migrate(h5_path: str) -> None:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Not found: {h5_path}")

    tmp_path = h5_path + ".migrated.tmp"

    with h5py.File(h5_path, "r") as src:
        n_rows = len(src["labels"])
        log.info("Source: %s  (%d rows)", h5_path, n_rows)

        runtime_maxshape = src["runtimes"].maxshape if "runtimes" in src else None
        if runtime_maxshape is not None and runtime_maxshape[1] is None:
            log.info("runtimes already has maxshape=(None, None) — nothing to do.")
            return

        with h5py.File(tmp_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            for key in sorted(src.keys()):
                _copy_dataset(key, src, dst)

            dst.flush()

    backup = h5_path + ".bak"
    os.rename(h5_path, backup)
    os.rename(tmp_path, h5_path)
    log.info("Migration complete. Original backed up to %s", backup)


if __name__ == "__main__":
    path = _resolve_path(sys.argv[1] if len(sys.argv) > 1 else None)
    migrate(path)
