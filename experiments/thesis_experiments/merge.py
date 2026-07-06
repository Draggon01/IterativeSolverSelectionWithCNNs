"""
merge.py — Merge multiple dataset.h5 files into a single output file.

All source files are opened read-only (h5py mode="r") and are never modified.
Datasets are concatenated along axis 0. Attributes are taken from the first source.

Compatibility requirements:
  - All sources must have the same solver list (attrs["solvers"]).
  - If "images" is present in all sources, image_mode and image_size must match.
  - Datasets absent from any source are skipped with a warning.

Environment variables:
  SRC_DIRS   Space-separated list of directories (each must contain dataset.h5)
             Default: /workspace/data/base /workspace/data/suitesparse_githubdata
  OUT_DIR    Output directory for the merged dataset.h5
             Default: /workspace/data/merged
  CHUNK      Rows per streaming batch (tune for memory vs speed)  (default 500)
"""

import logging
import os

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_DIRS = os.getenv(
    "SRC_DIRS",
    "/workspace/data/base /workspace/data/suitesparse_githubdata",
).split()
OUT_DIR = os.getenv("OUT_DIR", "/workspace/data/merged")
CHUNK   = int(os.getenv("CHUNK", "500"))


def _is_vlen(ds: h5py.Dataset) -> bool:
    return (
        h5py.check_vlen_dtype(ds.dtype) is not None
        or h5py.check_string_dtype(ds.dtype) is not None
    )


def _chunk_shape(chunk: int, item_shape: tuple) -> tuple:
    if not item_shape:
        return (min(chunk, 256),)
    return (min(chunk, 64), *item_shape)


def _copy_dataset(key: str, srcs: list, dst: h5py.File) -> int:
    """Stream-copy dataset `key` from all source files into dst. Returns total rows written."""
    example = srcs[0][key]
    vlen     = _is_vlen(example)
    item_shape = example.shape[1:]

    if vlen:
        ds = dst.create_dataset(
            key, shape=(0,), maxshape=(None,),
            dtype=example.dtype, chunks=(256,),
        )
    else:
        full_shape = (0, *item_shape) if item_shape else (0,)
        max_shape  = (None, *item_shape) if item_shape else (None,)
        ds = dst.create_dataset(
            key, shape=full_shape, maxshape=max_shape,
            dtype=example.dtype,
            chunks=_chunk_shape(CHUNK, item_shape),
        )

    offset = 0
    for src in srcs:
        src_ds = src[key]
        n = len(src_ds)
        for start in range(0, n, CHUNK):
            end   = min(start + CHUNK, n)
            batch = src_ds[start:end]
            count = end - start
            ds.resize(offset + count, axis=0)
            ds[offset : offset + count] = batch
            offset += count

    log.info("  %-16s  %d rows", key, offset)
    return offset


def main() -> None:
    src_paths = [
        d if d.endswith(".h5") else os.path.join(d, "dataset.h5")
        for d in SRC_DIRS
    ]
    out_path  = os.path.join(OUT_DIR, "dataset.h5")
    tmp_path  = out_path + ".tmp"

    for p in src_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Source not found: {p}")

    real_out = os.path.realpath(out_path)
    for p in src_paths:
        if os.path.realpath(p) == real_out:
            raise ValueError(f"Output path collides with a source file: {p}")

    log.info("Sources (%d):", len(src_paths))
    for p in src_paths:
        log.info("  %s", p)
    log.info("Output : %s", out_path)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Open every source read-only — files cannot be modified
    srcs = [h5py.File(p, "r") for p in src_paths]
    try:
        # ── Compatibility checks ──────────────────────────────────────────────
        solver_names = None
        for src, p in zip(srcs, src_paths):
            names = list(src.attrs.get("solvers", []))
            if solver_names is None:
                solver_names = names
            elif names != solver_names:
                raise ValueError(
                    f"Solver list mismatch:\n"
                    f"  {src_paths[0]}: {solver_names}\n"
                    f"  {p}: {names}"
                )
            log.info("  %s  —  %d samples", p, len(src["labels"]))

        total = sum(len(s["labels"]) for s in srcs)
        log.info("Total after merge: %d samples", total)

        # ── Determine which keys to merge ─────────────────────────────────────
        all_key_sets = [set(s.keys()) for s in srcs]
        common_keys  = set.intersection(*all_key_sets)
        partial_keys = set.union(*all_key_sets) - common_keys
        # mat_* vlen keys: carry from sources that have them; pad empty arrays
        # for sources that don't (render.py falls back to .mtx cache for those rows)
        mat_keys     = {k for k in partial_keys if k.startswith("mat_")}
        skipped_keys = partial_keys - mat_keys
        if skipped_keys:
            log.warning("Skipping (not in all sources): %s", sorted(skipped_keys))
        if mat_keys:
            log.info("Partial keys (will pad missing rows with empty arrays): %s",
                     sorted(mat_keys))

        if "images" in common_keys:
            modes = [src.attrs.get("image_mode", "?") for src in srcs]
            sizes = [src["images"].shape[1]             for src in srcs]
            if len(set(modes)) > 1 or len(set(sizes)) > 1:
                log.warning(
                    "image_mode/size mismatch across sources %s %s — skipping 'images'."
                    " Re-run render on the merged file to add images.",
                    modes, sizes,
                )
                common_keys.discard("images")

        # ── Write merged output ───────────────────────────────────────────────
        with h5py.File(tmp_path, "w") as dst:
            for k, v in srcs[0].attrs.items():
                dst.attrs[k] = v
            dst.attrs["merged_sources"] = [os.path.abspath(p) for p in src_paths]

            log.info("Merging %d common datasets …", len(common_keys))
            for key in sorted(common_keys):
                _copy_dataset(key, srcs, dst)

            # mat_* keys: copy from sources that have them, pad zeros/empty for others
            if mat_keys:
                log.info("Merging %d partial (mat_*) datasets with padding …", len(mat_keys))
                for key in sorted(mat_keys):
                    example_src = next(s for s in srcs if key in s)
                    example_ds  = example_src[key]
                    is_vlen     = _is_vlen(example_ds)
                    item_shape  = example_ds.shape[1:]

                    if is_vlen:
                        ds = dst.create_dataset(
                            key, shape=(0,), maxshape=(None,),
                            dtype=example_ds.dtype, chunks=(256,),
                        )
                    else:
                        full_shape = (0, *item_shape) if item_shape else (0,)
                        max_shape  = (None, *item_shape) if item_shape else (None,)
                        ds = dst.create_dataset(
                            key, shape=full_shape, maxshape=max_shape,
                            dtype=example_ds.dtype,
                            chunks=_chunk_shape(CHUNK, item_shape),
                        )

                    offset = 0
                    for src in srcs:
                        n = len(src["labels"])
                        if key in src:
                            src_ds = src[key]
                            for start in range(0, n, CHUNK):
                                end   = min(start + CHUNK, n)
                                batch = src_ds[start:end]
                                count = end - start
                                ds.resize(offset + count, axis=0)
                                ds[offset : offset + count] = batch
                                offset += count
                        else:
                            # Pad rows from sources that don't have this key
                            ds.resize(offset + n, axis=0)
                            if is_vlen:
                                inner = h5py.check_vlen_dtype(example_ds.dtype)
                                empty = np.array([], dtype=inner)
                                for i in range(n):
                                    ds[offset + i] = empty
                            else:
                                # zero-fill fixed-shape rows (e.g. mat_shape → (0,0))
                                pad = np.zeros((n, *item_shape), dtype=example_ds.dtype)
                                ds[offset : offset + n] = pad
                            offset += n
                    log.info("  %-16s  %d rows", key, offset)

            dst.flush()

        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)

    finally:
        for src in srcs:
            src.close()

    log.info("Done — wrote %d samples → %s", total, out_path)


if __name__ == "__main__":
    main()
