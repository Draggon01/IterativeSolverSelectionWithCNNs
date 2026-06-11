"""
mm_ingest.py — Download SuiteSparse matrices and append them to the baseline dataset.

Mirrors src/ingest_suitesparse.py but extracts MM-AutoSolver features
(17 numerical features, 128×128 density image) instead of the main
pipeline's format. Appends to the same dataset.h5 produced by mm_generate.py
so training uses both synthetic and real matrices.

Supports four modes (set via MODE env var):
  auto       (default) — query SuiteSparse via ssgetpy, download automatically
  manual               — scan a local directory (MTX_DIR) for .mtx / .mat files
  csv                  — download exactly the matrices listed in a CSV file
  githubdata           — download the hardcoded list of 621 benchmark matrices

Environment variables:
  MODE          "auto", "manual", "csv", or "githubdata"  (default auto)
  DATA_DIR      Path containing dataset.h5                (default ./data)
  CACHE_DIR     Local cache for downloads                 (default ./data/suitesparse_cache)
  MTX_DIR       Directory of .mtx files [manual]          (default ./data/mtx)
  CSV_FILE      Path to CSV file [csv mode]               (required for csv mode)
  CSV_NAME_COL  Column name for matrix names [csv]        (default matrix_name)
  MIN_N         Minimum matrix dimension                   (default 100)
  MAX_N         Maximum matrix dimension                   (default 50000)
  N_MATRICES    Max matrices to add                        (default 1000)
  ONLY_SPD      Restrict to SPD matrices                   (default 0 = all real square)
  SEED          RNG seed for RHS vectors                   (default 0)
  MAX_ITER      Max KSP iterations per solver              (default 2000)
  TOL           Convergence tolerance                      (default 1e-8)
"""

import glob
import logging
import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Reuse matrix loading and classification from the main pipeline
from ingest_suitesparse import load_matrix, classify, run_ksp

from mm_model import (
    mm_features, mm_density_image, MM_N_FEATURES, MM_IMAGE_SIZE,
    MM_SOLVER_PAIRS, MM_SOLVER_NAMES, MM_SOLVER_IDX, MM_N_SOLVERS, MM_APPLICABLE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODE         = os.getenv("MODE",         "auto")
DATA_DIR     = os.getenv("DATA_DIR",     "./data")
CACHE_DIR    = os.getenv("CACHE_DIR",    os.path.join(DATA_DIR, "suitesparse_cache"))
MTX_DIR      = os.getenv("MTX_DIR",      os.path.join(DATA_DIR, "mtx"))
CSV_FILE     = os.getenv("CSV_FILE",     "")
CSV_NAME_COL = os.getenv("CSV_NAME_COL", "matrix_name")
MIN_N        = int(os.getenv("MIN_N",        "100"))
MAX_N        = int(os.getenv("MAX_N",        "50000"))
N_MATRICES   = int(os.getenv("N_MATRICES",   "1000"))
ONLY_SPD     = os.getenv("ONLY_SPD",    "0") == "1"
SEED         = int(os.getenv("SEED",         "0"))
MAX_ITER     = int(os.getenv("MAX_ITER",     "2000"))
TOL          = float(os.getenv("TOL",        "1e-8"))


# ── hardcoded benchmark matrix list (621 matrices) ───────────────────────────

GITHUBDATA_MATRICES: tuple[str, ...] = (
    "1138_bus", "3elt", "3elt_dual", "ACTIVSg2000", "add20", "add32",
    "adder_dcop_01", "adder_dcop_02", "adder_dcop_03", "adder_dcop_04",
    "adder_dcop_05", "adder_dcop_06", "adder_dcop_07", "adder_dcop_08",
    "adder_dcop_09", "adder_dcop_10", "adder_dcop_11", "adder_dcop_12",
    "adder_dcop_13", "adder_dcop_14", "adder_dcop_15", "adder_dcop_16",
    "adder_dcop_17", "adder_dcop_18", "adder_dcop_19", "adder_dcop_20",
    "adder_dcop_21", "adder_dcop_22", "adder_dcop_23", "adder_dcop_24",
    "adder_dcop_25", "adder_dcop_26", "adder_dcop_27", "adder_dcop_28",
    "adder_dcop_29", "adder_dcop_30", "adder_dcop_31", "adder_dcop_32",
    "adder_dcop_33", "adder_dcop_34", "adder_dcop_35", "adder_dcop_36",
    "adder_dcop_37", "adder_dcop_38", "adder_dcop_39", "adder_dcop_40",
    "adder_dcop_41", "adder_dcop_42", "adder_dcop_43", "adder_dcop_44",
    "adder_dcop_45", "adder_dcop_46", "adder_dcop_47", "adder_dcop_48",
    "adder_dcop_49", "adder_dcop_50", "adder_dcop_51", "adder_dcop_52",
    "adder_dcop_53", "adder_dcop_54", "adder_dcop_55", "adder_dcop_56",
    "adder_dcop_57", "adder_dcop_58", "adder_dcop_59", "adder_dcop_60",
    "adder_dcop_61", "adder_dcop_62", "adder_dcop_63", "adder_dcop_64",
    "adder_dcop_65", "adder_dcop_66", "adder_dcop_67", "adder_dcop_68",
    "adder_dcop_69", "adder_trans_01", "adder_trans_02", "aft01",
    "airfoil1", "airfoil1_dual", "Alemdar", "as-735", "b2_ss", "barth",
    "barth4", "barth4-ones", "barth-ones", "bayer03", "bayer05", "bayer06",
    "bayer07", "bayer08", "bayer09", "bcspwr06", "bcspwr07", "bcspwr08",
    "bcspwr09", "bcspwr10", "bcsstk08", "bcsstk09", "bcsstk10", "bcsstk11",
    "bcsstk12", "bcsstk13", "bcsstk14", "bcsstk15", "bcsstk21", "bcsstk23",
    "bcsstk24", "bcsstk26", "bcsstk27", "bcsstm08", "bcsstm09", "bcsstm10",
    "bcsstm11", "bcsstm12", "bcsstm13", "bcsstm21", "bcsstm23", "bcsstm24",
    "bcsstm26", "bcsstm27", "bcsstm38", "b_dyn", "bibd_81_2", "bips98_1142",
    "bips98_606", "blckhole", "bwm2000", "c-18", "c-19", "c-20", "c-21",
    "c-22", "c-23", "c-24", "c-25", "c-26", "c-27", "c-28", "c-29", "c-30",
    "c-31", "c-32", "c-33", "c-34", "c-35", "c-36", "c-37", "c-38", "c-39",
    "c-40", "c-41", "cage8", "cage9", "CAG_mat1916", "ca-GrQc", "ca-HepTh",
    "California", "can_1054", "can_1072", "cavity05", "cavity06", "cavity07",
    "cavity08", "cavity09", "cavity10", "cavity11", "cavity12", "cavity13",
    "cavity14", "cavity15", "cavity16", "cavity17", "cavity18", "cavity19",
    "cavity20", "cavity21", "cavity22", "cavity23", "cavity24", "cavity25",
    "cavity26", "cegb3024", "cegb3306", "cell1", "cell2", "Chebyshev2",
    "Chebyshev3", "Chem97ZtZ", "circuit_1", "circuit_2", "circuit204",
    "coater1", "CollegeMsg", "commanche_dual", "comsol", "cryg10000",
    "cryg2500", "crystm01", "CSphd", "cz1268", "cz2548", "cz5108", "data",
    "delaunay_n10", "delaunay_n11", "delaunay_n12", "delaunay_n13", "diag",
    "dw1024", "dw2048", "dw4096", "dw8192", "dwt_1005", "dwt_1007",
    "dwt_1242", "dwt_2680", "dynamicSoaringProblem_2", "dynamicSoaringProblem_3",
    "dynamicSoaringProblem_4", "dynamicSoaringProblem_5", "dynamicSoaringProblem_6",
    "dynamicSoaringProblem_7", "dynamicSoaringProblem_8", "email",
    "email-Eu-core", "email-Eu-core-temporal", "EPA", "epb0", "Erdos02",
    "Erdos972", "Erdos982", "Erdos992", "eris1176", "eurqsa", "EVA", "ex10",
    "ex10hs", "ex12", "ex13", "ex14", "ex15", "ex18", "ex20", "ex23", "ex24",
    "ex26", "ex28", "ex29", "ex3", "EX3", "ex31", "ex32", "ex33", "ex36",
    "ex37", "ex4", "EX4", "ex6", "ex7", "ex8", "ex9", "extr1", "extr1b",
    "fd12", "filter2D", "flowmeter0", "flowmeter5", "fpga_dcop_01",
    "fpga_dcop_02", "fpga_dcop_03", "fpga_dcop_04", "fpga_dcop_05",
    "fpga_dcop_06", "fpga_dcop_07", "fpga_dcop_08", "fpga_dcop_09",
    "fpga_dcop_10", "fpga_dcop_11", "fpga_dcop_12", "fpga_dcop_13",
    "fpga_dcop_14", "fpga_dcop_15", "fpga_dcop_16", "fpga_dcop_17",
    "fpga_dcop_18", "fpga_dcop_19", "fpga_dcop_20", "fpga_dcop_21",
    "fpga_dcop_22", "fpga_dcop_23", "fpga_dcop_24", "fpga_dcop_25",
    "fpga_dcop_26", "fpga_dcop_27", "fpga_dcop_28", "fpga_dcop_29",
    "fpga_dcop_30", "fpga_dcop_31", "fpga_dcop_32", "fpga_dcop_33",
    "fpga_dcop_34", "fpga_dcop_35", "fpga_dcop_36", "fpga_dcop_37",
    "fpga_dcop_38", "fpga_dcop_39", "fpga_dcop_40", "fpga_dcop_41",
    "fpga_dcop_42", "fpga_dcop_43", "fpga_dcop_44", "fpga_dcop_45",
    "fpga_dcop_46", "fpga_dcop_47", "fpga_dcop_48", "fpga_dcop_49",
    "fpga_dcop_50", "fpga_dcop_51", "fpga_trans_01", "fpga_trans_02",
    "freeFlyingRobot_10", "freeFlyingRobot_11", "freeFlyingRobot_12",
    "freeFlyingRobot_13", "freeFlyingRobot_14", "freeFlyingRobot_15",
    "freeFlyingRobot_16", "freeFlyingRobot_2", "freeFlyingRobot_3",
    "freeFlyingRobot_4", "freeFlyingRobot_5", "freeFlyingRobot_6",
    "freeFlyingRobot_7", "freeFlyingRobot_8", "freeFlyingRobot_9",
    "fv1", "fv2", "fv3", "G22", "G23", "G24", "G25", "G26", "G27", "G28",
    "G29", "G30", "G31", "G32", "G33", "G34", "G35", "G36", "G37", "G38",
    "G39", "G40", "G41", "G42", "G43", "G44", "G45", "G46", "G47", "G48",
    "G49", "G50", "G51", "G52", "G53", "G54", "G55", "G56", "G57", "G58",
    "G59", "G60", "G61", "G62", "G63", "G64", "G65", "G66", "G67",
    "g7jac010", "g7jac010sc", "g7jac020", "g7jac020sc", "garon1",
    "GD06_Java", "GD96_a", "gemat11", "gemat12", "geom", "Goodwin_010",
    "Goodwin_013", "Goodwin_017", "Goodwin_023", "gre_1107", "grid2",
    "grid2_dual", "Hamrle2", "hangGlider_2", "hep-th", "hydr1", "hydr1c",
    "init_adder1", "iprob", "jagmesh2", "jagmesh3", "jagmesh4", "jagmesh5",
    "jagmesh6", "jagmesh7", "jagmesh8", "jagmesh9", "jan99jac020",
    "jan99jac020sc", "Kaufhold", "kineticBatchReactor_1",
    "kineticBatchReactor_2", "kineticBatchReactor_3", "kineticBatchReactor_4",
    "kineticBatchReactor_5", "kineticBatchReactor_6", "kineticBatchReactor_7",
    "kineticBatchReactor_8", "kineticBatchReactor_9", "Kohonen", "laser",
    "Lederberg", "LeGresley_2508", "LeGresley_4908", "lhr01", "lhr02",
    "lhr04", "lhr04c", "lhr07", "lhr07c", "lns_3937", "lnsp3937",
    "lock1074", "lock2232", "lock3491", "lowThrust_2", "lowThrust_3",
    "lshp1009", "lshp1270", "lshp1561", "lshp1882", "lshp2233", "lshp2614",
    "lshp3025", "lshp3466", "lung1", "M20PI_n", "M20PI_n1", "M40PI_n",
    "M40PI_n1", "M80PI_n", "M80PI_n1", "mahindas", "mark3jac020",
    "mark3jac020sc", "meg1", "meg4", "mhd3200a", "mhd3200b", "mhd4800a",
    "mhd4800b", "minnesota", "MISKnowledgeMap", "msc01050", "msc01440",
    "msc04515", "Muu", "mycielskian11", "n3c6-b7", "nasa1824",
    "nasa1824-perturbed", "nasa2146", "nasa2910", "nasa2910-nz", "nasa4704",
    "nasa4704-nz", "netscience", "netz4504", "nnc1374", "NotreDame_yeast",
    "ODLIS", "olm1000", "olm2000", "olm5000", "orani678", "orsirr_1",
    "orsreg_1", "p2p-Gnutella05", "p2p-Gnutella06", "p2p-Gnutella08",
    "p2p-Gnutella09", "Pd", "pde2961", "piston", "plat1919", "plbuckle",
    "plsk1919", "polblogs", "poli", "pores_2", "power", "qh1484", "radfr1",
    "raefsky5", "raefsky6", "rail_1357", "rail_5177", "rajat01", "rajat02",
    "rajat03", "rajat04", "rajat12", "rajat13", "rajat19", "rdb1250",
    "rdb1250l", "rdb2048", "rdb2048_noL", "rdb3200l", "rdb5000", "rdist1",
    "rdist2", "rdist3a", "reorientation_2", "reorientation_3",
    "reorientation_4", "reorientation_5", "reorientation_6", "reorientation_7",
    "reorientation_8", "Roget", "rw5151", "S20PI_n", "S20PI_n1", "S40PI_n",
    "S40PI_n1", "S80PI_n", "S80PI_n1", "saylr3", "saylr4", "SciMet",
    "sherman1", "sherman2", "sherman3", "sherman4", "sherman5", "shermanACa",
    "shermanACd", "shyy41", "Sieber", "SiH4", "SiNa", "SmaGri",
    "soc-sign-bitcoin-alpha", "soc-sign-bitcoin-otc", "spaceShuttleEntry_2",
    "spaceShuttleEntry_3", "spaceShuttleEntry_4", "spaceStation_10",
    "spaceStation_11", "spaceStation_12", "spaceStation_13", "spaceStation_14",
    "spaceStation_5", "spaceStation_6", "spaceStation_7", "spaceStation_8",
    "spaceStation_9", "spiral", "sstmodel", "sts4098", "stufe", "swang1",
    "swang2", "t2dal", "t2dal_a", "t2dal_bci", "t2dal_e", "t2d_q4",
    "t2d_q9", "thermal", "tols1090", "tols2000", "tols4000",
    "Trefethen_2000", "TS", "TSOPF_FS_b9_c1", "TSOPF_RS_b9_c6", "tub1000",
    "uk", "ukerbe1", "ukerbe1_dual", "USpowerGrid", "utm1700b", "utm3060",
    "utm5940", "viscoplastic1", "vsp_data_and_seymourl", "wang1", "wang2",
    "watt_1", "watt_2", "wb-cs-stanford", "west1505", "west2021",
    "whitaker3", "wiki-Vote", "yeast", "zenios", "Zewail",
)


# ── solver benchmarking ───────────────────────────────────────────────────────

def benchmark(A: sp.csr_matrix, b: np.ndarray, mat_type: str) -> "tuple[int | None, np.ndarray]":
    """Run the paper's 19 solver pairs; return (best_label, runtimes)."""
    all_times  = np.full(MM_N_SOLVERS, np.nan, dtype=np.float32)
    converged: dict[tuple, float] = {}

    for pair in MM_APPLICABLE[mat_type]:
        ksp_type, pc_type = pair
        ok, iters, t = run_ksp(A, b, ksp_type, pc_type)
        if ok:
            converged[pair] = t
            all_times[MM_SOLVER_IDX[pair]] = float(t)
        log.debug("  %-8s+%-8s  ok=%-5s  iters=%-4d  t=%.4fs",
                  ksp_type, pc_type, ok, iters, t)

    if not converged:
        return None, all_times

    best_pair = min(converged, key=converged.__getitem__)
    return int(MM_SOLVER_IDX[best_pair]), all_times


# ── HDF5 helpers ──────────────────────────────────────────────────────────────

def open_or_create_dataset(path: str) -> h5py.File:
    """Open dataset.h5 in append mode, creating all required datasets if absent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = h5py.File(path, "a")

    def _ensure(name, **kwargs):
        if name not in f:
            f.create_dataset(name, **kwargs)

    _ensure("images",   shape=(0, MM_IMAGE_SIZE, MM_IMAGE_SIZE),
            maxshape=(None, MM_IMAGE_SIZE, MM_IMAGE_SIZE),
            dtype="f4", chunks=(64, MM_IMAGE_SIZE, MM_IMAGE_SIZE))
    _ensure("features", shape=(0, MM_N_FEATURES), maxshape=(None, MM_N_FEATURES),
            dtype="f4", chunks=(256, MM_N_FEATURES))
    _ensure("labels",   shape=(0,), maxshape=(None,), dtype="i4", chunks=(256,))
    _ensure("runtimes", shape=(0, MM_N_SOLVERS), maxshape=(None, MM_N_SOLVERS),
            dtype="f4", chunks=(256, MM_N_SOLVERS))
    _ensure("source",   shape=(0,), maxshape=(None,),
            dtype=h5py.string_dtype(), chunks=(256,))

    if "solvers"    not in f.attrs: f.attrs["solvers"]    = MM_SOLVER_NAMES
    if "n_features" not in f.attrs: f.attrs["n_features"] = MM_N_FEATURES
    if "image_size" not in f.attrs: f.attrs["image_size"] = MM_IMAGE_SIZE

    return f


def already_ingested(f: h5py.File) -> set[str]:
    if "source" not in f or len(f["source"]) == 0:
        return set()
    return {s.decode() if isinstance(s, bytes) else s for s in f["source"][:]}


def append_sample(f: h5py.File, A: sp.csr_matrix, label: int,
                  times: np.ndarray, source: str) -> None:
    n = len(f["labels"])
    for ds in ("images", "features", "labels", "runtimes", "source"):
        f[ds].resize(n + 1, axis=0)
    f["images"][n]   = mm_density_image(A)
    f["features"][n] = mm_features(A)
    f["labels"][n]   = label
    f["runtimes"][n] = times
    f["source"][n]   = source


# ── ingestion pipeline ────────────────────────────────────────────────────────

def ingest_matrix(f: h5py.File, A: sp.csr_matrix, source: str,
                  isspd: bool, issym: bool, rng: np.random.Generator) -> bool:
    mat_type = classify(A, isspd, issym)
    b        = rng.standard_normal(A.shape[0])

    log.info("  Benchmarking %s  n=%d  nnz=%d  type=%s",
             source, A.shape[0], A.nnz, mat_type)

    label, times = benchmark(A, b, mat_type)
    if label is None:
        log.warning("  No solver converged for %s — skipping.", source)
        return False

    append_sample(f, A, label, times, source)
    f.flush()  # write to disk immediately so a crash doesn't lose data
    log.info("  Saved  best=%s", MM_SOLVER_NAMES[label])
    return True


# ── auto mode (ssgetpy) ───────────────────────────────────────────────────────

def run_auto(f: h5py.File, rng: np.random.Generator) -> None:
    try:
        import ssgetpy
    except ImportError:
        log.error("ssgetpy not installed. Run: pip install ssgetpy")
        return

    done    = already_ingested(f)
    saved   = skipped = 0

    search_kwargs: dict = dict(limit=N_MATRICES * 5)
    if ONLY_SPD:
        search_kwargs["isspd"] = True

    log.info("Querying SuiteSparse collection ...")
    try:
        results = ssgetpy.search(**search_kwargs)
    except Exception as exc:
        log.error("SuiteSparse query failed: %s", exc)
        return

    # Filter by size, square shape, and real type using metadata (before any download)
    results = [m for m in results
               if MIN_N <= m.rows <= MAX_N
               and m.rows == m.cols
               and getattr(m, 'dtype', 'real') == 'real']
    log.info("Found %d candidate matrices after filtering (n=[%d,%d]).",
             len(results), MIN_N, MAX_N)
    os.makedirs(CACHE_DIR, exist_ok=True)

    for matrix in results:
        if saved >= N_MATRICES:
            break

        source = f"suitesparse/{matrix.group}/{matrix.name}"
        if source in done:
            log.info("Already ingested %s — skipping.", source)
            saved += 1
            continue

        try:
            matrix.download(destpath=CACHE_DIR, format="MM", extract=True)
        except Exception as exc:
            log.warning("Download failed for %s: %s", source, exc)
            skipped += 1
            continue

        hits = glob.glob(os.path.join(CACHE_DIR, matrix.name, "*.mtx"))
        if not hits:
            log.warning("No .mtx found for %s", source)
            skipped += 1
            continue

        A = load_matrix(hits[0])
        if A is None:
            skipped += 1
            continue

        issym = bool(matrix.isspd) or (getattr(matrix, 'psym', 0) == 1 and getattr(matrix, 'nsym', 0) == 1)
        ok = ingest_matrix(f, A, source, isspd=bool(matrix.isspd),
                           issym=issym, rng=rng)
        if ok:
            saved += 1
        else:
            skipped += 1

    log.info("Auto mode complete: saved=%d  skipped=%d", saved, skipped)


# ── manual mode ───────────────────────────────────────────────────────────────

def run_manual(f: h5py.File, rng: np.random.Generator) -> None:
    mtx_files = sorted(
        glob.glob(os.path.join(MTX_DIR, "**", "*.mtx"), recursive=True) +
        glob.glob(os.path.join(MTX_DIR, "**", "*.mat"), recursive=True)
    )
    if not mtx_files:
        log.error("No .mtx or .mat files found under %s", MTX_DIR)
        return

    done    = already_ingested(f)
    saved   = skipped = 0

    for mtx_path in mtx_files:
        if saved >= N_MATRICES:
            break

        rel    = os.path.relpath(mtx_path, MTX_DIR)
        source = "manual/" + rel.replace(os.sep, "/").removesuffix(".mtx")

        if source in done:
            log.info("Already ingested %s — skipping.", source)
            saved += 1
            continue

        A = load_matrix(mtx_path)
        if A is None:
            skipped += 1
            continue

        ok = ingest_matrix(f, A, source, isspd=False, issym=False, rng=rng)
        if ok:
            saved += 1
        else:
            skipped += 1

    log.info("Manual mode complete: saved=%d  skipped=%d", saved, skipped)


# ── shared name-based download helper ────────────────────────────────────────

def _ingest_by_names(f: h5py.File, names: list[str], rng: np.random.Generator,
                     label: str) -> None:
    """Download and benchmark each matrix in `names` from SuiteSparse."""
    try:
        import ssgetpy
    except ImportError:
        log.error("ssgetpy not installed. Run: pip install ssgetpy")
        return

    done    = already_ingested(f)
    saved   = skipped = 0
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name in names:
        if saved >= N_MATRICES:
            break

        if any(s.endswith(f"/{name}") for s in done):
            log.info("Already ingested %s — skipping.", name)
            saved += 1
            continue

        try:
            results = ssgetpy.search(name=name, limit=10)
        except Exception as exc:
            log.warning("ssgetpy search failed for %s: %s", name, exc)
            skipped += 1
            continue

        matrix = next((m for m in results if m.name == name), None)
        if matrix is None:
            log.warning("Matrix '%s' not found in SuiteSparse — skipping.", name)
            skipped += 1
            continue

        # Use metadata only for size pre-filter (saves bandwidth); squareness
        # is checked after loading because ssgetpy metadata can be unreliable.
        n = getattr(matrix, 'rows', 0)
        if n and not (MIN_N <= n <= MAX_N):
            log.info("Skipping %s (n=%d outside [%d, %d]).", name, n, MIN_N, MAX_N)
            skipped += 1
            continue

        try:
            matrix.download(destpath=CACHE_DIR, format="MM", extract=True)
        except Exception as exc:
            log.warning("Download failed for %s: %s", name, exc)
            skipped += 1
            continue

        hits = glob.glob(os.path.join(CACHE_DIR, name, "*.mtx"))
        if not hits:
            log.warning("No .mtx file found for %s after download.", name)
            skipped += 1
            continue

        # Prefer {name}.mtx over auxiliary files (e.g. coord, rhs, b vectors)
        exact = os.path.join(CACHE_DIR, name, f"{name}.mtx")
        mtx_path = exact if os.path.isfile(exact) else hits[0]

        A = load_matrix(mtx_path, require_nonzero_diag=False)
        if A is None:
            skipped += 1
            continue

        if A.shape[0] != A.shape[1]:
            log.info("Skipping %s (not square after load: %d×%d).", name, *A.shape)
            skipped += 1
            continue

        issym = bool(matrix.isspd) or (
            getattr(matrix, 'psym', 0) == 1 and getattr(matrix, 'nsym', 0) == 1
        )
        source = f"suitesparse/{name}"
        ok = ingest_matrix(f, A, source, isspd=bool(matrix.isspd), issym=issym, rng=rng)
        if ok:
            saved += 1
        else:
            skipped += 1

    log.info("%s complete: saved=%d  skipped=%d", label, saved, skipped)


# ── csv mode ──────────────────────────────────────────────────────────────────

def run_csv(f: h5py.File, rng: np.random.Generator) -> None:
    import csv as csv_mod

    if not CSV_FILE:
        log.error("CSV_FILE env var not set — required for csv mode.")
        return
    if not os.path.isfile(CSV_FILE):
        log.error("CSV_FILE not found: %s", CSV_FILE)
        return

    with open(CSV_FILE, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        names = list(dict.fromkeys(
            row[CSV_NAME_COL].strip()
            for row in reader
            if row.get(CSV_NAME_COL, "").strip()
        ))

    log.info("CSV mode — %d unique names from %s (col=%s)", len(names), CSV_FILE, CSV_NAME_COL)
    _ingest_by_names(f, names, rng, label="CSV mode")


# ── githubdata mode ───────────────────────────────────────────────────────────

def run_githubdata(f: h5py.File, rng: np.random.Generator) -> None:
    log.info("Githubdata mode — %d hardcoded benchmark matrices.", len(GITHUBDATA_MATRICES))
    _ingest_by_names(f, list(GITHUBDATA_MATRICES), rng, label="Githubdata mode")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    from petsc4py import PETSc
    PETSc.Options()["ksp_error_if_not_converged"] = False

    h5_path = os.path.join(DATA_DIR, "dataset.h5")
    rng     = np.random.default_rng(SEED)

    with open_or_create_dataset(h5_path) as f:
        n_before = len(f["labels"])
        log.info("Dataset at %s — %d existing samples.", h5_path, n_before)

        if MODE == "manual":
            log.info("Manual mode — scanning %s", MTX_DIR)
            run_manual(f, rng)
        elif MODE == "csv":
            log.info("CSV mode — %s  col=%s", CSV_FILE, CSV_NAME_COL)
            run_csv(f, rng)
        elif MODE == "githubdata":
            run_githubdata(f, rng)
        else:
            log.info("Auto mode — querying SuiteSparse (n=[%d,%d]  max=%d)",
                     MIN_N, MAX_N, N_MATRICES)
            run_auto(f, rng)

        n_after = len(f["labels"])
        log.info("Done. Added %d SuiteSparse samples (total=%d).",
                 n_after - n_before, n_after)


if __name__ == "__main__":
    main()
