"""
drop_rows.py — Remove specific rows from dataset.h5 by source string.

Rows whose 'source' value contains any of the given substrings are dropped.
All other rows are streamed to a new file; the original is kept as .bak.

Usage:
    python drop_rows.py <substring> [<substring> ...]

Examples:
    python drop_rows.py suitesparse/Hollinger/mark3jac040sc
    python drop_rows.py mark3jac040sc mark3jac060 mcca mesh2e1 mesh2em5

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


def _is_vlen(ds: h5py.Dataset) -> bool:
    return (
        h5py.check_vlen_dtype(ds.dtype) is not None
        or h5py.check_string_dtype(ds.dtype) is not None
    )


def drop_rows(h5_path: str, substrings: list[str]) -> None:
    if not substrings:
        print("No substrings given — nothing to do.")
        return

    tmp_path = h5_path + ".drop.tmp"
    bak_path = h5_path + ".bak"

    with h5py.File(h5_path, "r") as src:
        sources = [
            (s.decode() if isinstance(s, bytes) else s)
            for s in src["source"][:]
        ]
        n_total = len(sources)

        keep_mask = np.array([
            not any(sub in s for sub in substrings)
            for s in sources
        ])
        keep_idx  = np.where(keep_mask)[0]
        drop_idx  = np.where(~keep_mask)[0]

        if len(drop_idx) == 0:
            log.info("No matching rows found for: %s", substrings)
            return

        log.info("Dropping %d / %d rows:", len(drop_idx), n_total)
        for i in drop_idx:
            log.info("  [%d] %s", i, sources[i])

        with h5py.File(tmp_path, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            for key in sorted(src.keys()):
                src_ds     = src[key]
                item_shape = src_ds.shape[1:]
                vlen       = _is_vlen(src_ds)
                n_keep     = len(keep_idx)

                if vlen:
                    dst_ds = dst.create_dataset(
                        key, shape=(n_keep,), maxshape=(None,),
                        dtype=src_ds.dtype, chunks=(256,),
                    )
                elif key == "runtimes":
                    dst_ds = dst.create_dataset(
                        key, shape=(n_keep, *item_shape), maxshape=(None, None),
                        dtype=src_ds.dtype, chunks=(min(CHUNK, 256), *item_shape),
                    )
                elif item_shape:
                    dst_ds = dst.create_dataset(
                        key, shape=(n_keep, *item_shape), maxshape=(None, *item_shape),
                        dtype=src_ds.dtype, chunks=(min(CHUNK, 64), *item_shape),
                    )
                else:
                    dst_ds = dst.create_dataset(
                        key, shape=(n_keep,), maxshape=(None,),
                        dtype=src_ds.dtype, chunks=(256,),
                    )

                out = 0
                for start in range(0, n_total, CHUNK):
                    end     = min(start + CHUNK, n_total)
                    batch_i = keep_idx[(keep_idx >= start) & (keep_idx < end)] - start
                    if len(batch_i) == 0:
                        continue
                    batch  = src_ds[start:end][batch_i]
                    count  = len(batch_i)
                    dst_ds[out : out + count] = batch
                    out += count

                log.info("  %-16s  %d rows kept", key, n_keep)

            dst.flush()

    os.rename(h5_path, bak_path)
    os.rename(tmp_path, h5_path)
    log.info("Done. %d rows removed. Original backed up to %s", len(drop_idx), bak_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = os.path.join(DATA_DIR, "dataset.h5")
    drop_rows(path, sys.argv[1:])
