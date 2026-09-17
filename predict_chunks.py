import os
import sys
import json
import time
import shutil
import traceback
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import torch

# --- DIRECTORY SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
ARTIFACTS_DIR = os.environ.get(
    "NDVI_ARTIFACTS_DIR",
    os.environ.get(
        "NDVI_ARCHIVE_DIR",
        os.path.join(BASE_DIR, "GHMC_Dekadal_v2"),
    ),
)
CACHE_DIR = os.environ.get(
    "NDVI_CACHE_DIR",
    os.path.join(BASE_DIR, "cache"),
)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- PRESTO IMPORT ---
# Use single_file_presto only. Importing presto.dataops pulls hurry.filesize,
# openmapflow, and Earth Engine -- none of which inference needs.
sys.path.append(os.path.join(BASE_DIR, "presto"))
import single_file_presto as presto_pkg

# Band list and (shift, scale) copied from nasaharvest/presto
# presto/dataops/pipelines/s1_s2_era5_srtm.py
S1_BANDS = ["VV", "VH"]
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
S2_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
SRTM_BANDS = ["elevation", "slope"]
DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
REMOVED_BANDS = ["B1", "B10"]
BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + SRTM_BANDS + ["NDVI"]
ADD_BY = (
    [(S1_SHIFT_VALUES + S2_SHIFT_VALUES + ERA5_SHIFT_VALUES)[i]
     for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + [0.0, 0.0]
    + [0.0]
)
DIVIDE_BY = (
    [(S1_DIV_VALUES + S2_DIV_VALUES + ERA5_DIV_VALUES)[i]
     for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + [2000.0, 50.0]
    + [1.0]
)
ADD_BY_NP = np.asarray(ADD_BY, dtype=np.float32)
DIVIDE_BY_NP = np.asarray(DIVIDE_BY, dtype=np.float32)
NORMED_BANDS = [x for x in BANDS if x != "B9"]
KEEP_BANDS = [i for i, val in enumerate(BANDS) if val != "B9"]
NDVI_NORM_IDX = NORMED_BANDS.index("NDVI")
B8_IDX = NORMED_BANDS.index("B8")
B4_IDX = NORMED_BANDS.index("B4")


def _calculate_ndvi(input_array):
    if input_array.ndim == 2:
        n, d = input_array[:, B8_IDX], input_array[:, B4_IDX]
    elif input_array.ndim == 3:
        n, d = input_array[:, :, B8_IDX], input_array[:, :, B4_IDX]
    else:
        raise ValueError(f"Expected 2 or 3 dims, got {input_array.ndim}")
    denom = n + d
    if isinstance(n, np.ndarray):
        out = np.zeros(n.shape, dtype=np.float32)
        np.divide(n - d, denom, out=out, where=denom > 0)
        return out
    return torch.where(denom > 0, (n - d) / denom, 0)


def NORMALIZE_FN(x):
    if isinstance(x, np.ndarray):
        x = ((x + ADD_BY_NP) / DIVIDE_BY_NP).astype(np.float32)
    else:
        x = (x + torch.tensor(ADD_BY)) / torch.tensor(DIVIDE_BY)
    if x.ndim == 2:
        x = x[:, KEEP_BANDS]
        x[:, NDVI_NORM_IDX] = _calculate_ndvi(x)
    else:
        x = x[:, :, KEEP_BANDS]
        x[:, :, NDVI_NORM_IDX] = _calculate_ndvi(x)
    return x


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def tif(prefix, label=None):
    fname = f"{prefix}_{label}.tif" if label else f"{prefix}.tif"
    return os.path.join(DOWNLOADS_DIR, fname)


def _select_device(kind):
    kind = (kind or "auto").lower()
    if kind == "cpu":
        return torch.device("cpu")
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if kind == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but is not available")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Presto's boolean masking is slower on torch 2.1 MPS than on M4 CPU.
    return torch.device("cpu")


def _pixel_xy(transform, rows, cols):
    """Pixel-center coords from an affine transform. Matches rasterio.transform.xy."""
    cols_f = cols + 0.5
    rows_f = rows + 0.5
    xs = transform.a * cols_f + transform.b * rows_f + transform.c
    ys = transform.d * cols_f + transform.e * rows_f + transform.f
    return xs, ys


def _rowcol_floor(transform, xs, ys, height, width):
    """Vectorized rasterio.transform.rowcol (default op=numpy.floor)."""
    inv = ~transform
    cols_f = inv.a * xs + inv.b * ys + inv.c
    rows_f = inv.d * xs + inv.e * ys + inv.f
    er = np.clip(np.floor(rows_f).astype(np.int64), 0, height - 1)
    ec = np.clip(np.floor(cols_f).astype(np.int64), 0, width - 1)
    return er, ec


def _fill_mask(mask, invalid, t, cols):
    if not np.any(invalid):
        return
    mask[np.flatnonzero(invalid)[:, None], t, cols] = 1


def token_counts(mask_np, dw_np, dw_missing):
    c = np.zeros(mask_np.shape[0], dtype=np.int64)
    for name, idxs in BANDS_GROUPS_IDX.items():
        g = mask_np[:, :, idxs].max(axis=-1) > 0
        c += (g[:, :1] if name == "SRTM" else g).sum(axis=1)
    c += (dw_np == dw_missing).sum(axis=1)
    return c


BANDS_GROUPS_IDX = OrderedDict([
    ("S1", [0, 1]), ("S2_RGB", [2, 3, 4]), ("S2_Red_Edge", [5, 6, 7]),
    ("S2_NIR_10m", [8]), ("S2_NIR_20m", [9]), ("S2_SWIR", [10, 11]),
    ("ERA5", [12, 13]), ("SRTM", [14, 15]), ("NDVI", [16])
])


def _configure_threads(n_threads):
    n_threads = max(1, int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass


def _run_worker(cfg):
    try:
        _run_worker_impl(cfg)
    except Exception:
        rank = cfg.get("rank", "?")
        print(f"Worker {rank} crashed:", flush=True)
        traceback.print_exc()
        raise


def _run_worker_impl(cfg):
    rank = cfg["rank"]
    n_workers = cfg["n_workers"]
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    _configure_threads(cfg["cpu_threads"])
    device = torch.device(cfg["device"])

    ARCHIVE = cfg["archive"]
    DEKADS = ARCHIVE["dekads"]
    N_DEKADS = ARCHIVE["n_dekads"]
    DEKAD_DAYS = ARCHIVE["dekad_days"]
    NODATA_I16 = ARCHIVE["nodata_i16"]
    S1_SCALE = ARCHIVE["s1_scale"]
    SLOPE_SCALE = ARCHIVE["slope_scale"]
    DW_MISSING = ARCHIVE["dw_missing_class"]
    ARCHIVE_S2_BANDS = ARCHIVE["s2_bands"]
    T = ARCHIVE["context_length"]
    TARGET_T = T - 1
    WINDOW = DEKADS[N_DEKADS - T: N_DEKADS]
    MONTHS_WINDOW = np.array([d["month_index"] for d in WINDOW], dtype=np.int64)

    RAW_IDX = {b: i for i, b in enumerate(BANDS)}
    S2_RAW_IDXS = [RAW_IDX[b] for b in ARCHIVE_S2_BANDS]
    S1_RAW_IDXS = [RAW_IDX["VV"], RAW_IDX["VH"]]
    ERA5_RAW_IDXS = [RAW_IDX["temperature_2m"], RAW_IDX["total_precipitation"]]
    SRTM_RAW_IDXS = [RAW_IDX["elevation"], RAW_IDX["slope"]]
    N_IDX = {b: i for i, b in enumerate(NORMED_BANDS)}
    NDVI_IDX = N_IDX["NDVI"]
    S1_COLS = np.array([N_IDX["VV"], N_IDX["VH"]], dtype=np.int64)
    SRTM_COLS = np.array([N_IDX["elevation"], N_IDX["slope"]], dtype=np.int64)
    S2_COLS = [N_IDX[b] for b in ARCHIVE_S2_BANDS]
    OPTICAL_COLS = np.array(sorted(S2_COLS + [NDVI_IDX]), dtype=np.int64)

    model = presto_pkg.Presto.construct()
    model.load_state_dict(torch.load(cfg["weights_path"], map_location=device))
    model = model.to(device).eval()
    if rank == 0:
        print(f"Model loaded on {device}.", flush=True)

    hs = {}
    for d in WINDOW:
        for pfx in ("S2", "S2OBS", "S1", "S1OBS", "DW"):
            hs[(pfx, d["label"])] = rasterio.open(tif(pfx, d["label"]))
    srtm_h = rasterio.open(tif("SRTM_Static"))

    era5_arrs = []
    era5_transform = era5_h = era5_w = None
    for d in WINDOW:
        with rasterio.open(tif("ERA5", d["label"])) as src:
            arr = src.read()[:2]
            if era5_transform is None:
                era5_transform, era5_h, era5_w = src.transform, src.height, src.width
            era5_arrs.append(arr)
    era5 = np.stack(era5_arrs).astype(np.float32)

    W = cfg["width"]
    H = cfg["height"]
    ref_transform = rasterio.Affine(*cfg["transform"])
    to_wgs84 = Transformer.from_crs(cfg["crs_wkt"], "EPSG:4326", always_xy=True)

    ROWS_PER_CHUNK = cfg["rows_per_chunk"]
    BATCH_SIZE = cfg["batch_size"]
    TOTAL_CHUNKS = cfg["total_chunks"]
    CHUNKS_DIR = cfg["chunks_dir"]
    io_workers = min(8, T)

    assigned = [
        cid for cid in range(rank, TOTAL_CHUNKS, n_workers)
        if not os.path.exists(os.path.join(CHUNKS_DIR, f"chunk_{cid:05d}.npz"))
    ]
    n_done = 0
    t_all = time.perf_counter()
    print(f"Worker {rank}: {len(assigned)} chunks to compute", flush=True)

    def load_timestep(t_d):
        t, d = t_d
        lbl = d["label"]
        refl = hs[("S2", lbl)].read(window=win).astype(np.float32)
        obs = hs[("S2OBS", lbl)].read(window=win)
        bs = hs[("S1", lbl)].read(window=win).astype(np.float32)
        s1obs = hs[("S1OBS", lbl)].read(window=win)
        dw_t = hs[("DW", lbl)].read(window=win)[0]
        return t, refl, obs, bs, s1obs, dw_t

    try:
        with torch.inference_mode():
            for cid in assigned:
                c_path = os.path.join(CHUNKS_DIR, f"chunk_{cid:05d}.npz")
                t0 = time.perf_counter()
                r0 = cid * ROWS_PER_CHUNK
                r1 = min(r0 + ROWS_PER_CHUNK, H)
                ch = r1 - r0
                win = Window(0, r0, W, ch)
                npx = ch * W

                x_raw = np.zeros((npx, T, len(BANDS)), dtype=np.float32)
                mask = np.zeros((npx, T, len(NORMED_BANDS)), dtype=np.float32)
                dw = np.full((npx, T), DW_MISSING, dtype=np.int64)
                s2_valid = np.zeros((npx, T), dtype=bool)
                s2_age = np.full((npx, T), 255, dtype=np.uint8)

                tag = f"w{rank} " if n_workers > 1 else ""
                print(f"{tag}Chunk {cid + 1}/{TOTAL_CHUNKS}: reading rasters...", flush=True)
                with ThreadPoolExecutor(max_workers=io_workers) as pool:
                    loaded = list(pool.map(load_timestep, enumerate(WINDOW)))

                for t, refl, obs, bs, s1obs, dw_t in loaded:
                    nobs = obs[0].reshape(npx)
                    s2_age[:, t] = obs[1].reshape(npx)
                    valid = nobs > 0
                    s2_valid[:, t] = valid
                    refl = np.where(refl == NODATA_I16, 0.0, refl)
                    x_raw[:, t, S2_RAW_IDXS] = refl.reshape(len(S2_RAW_IDXS), npx).T
                    _fill_mask(mask, ~valid, t, OPTICAL_COLS)

                    bs = np.where(bs == NODATA_I16, 0.0, bs) / S1_SCALE
                    x_raw[:, t, S1_RAW_IDXS] = bs.reshape(2, npx).T
                    _fill_mask(mask, ~(s1obs[0].reshape(npx) > 0), t, S1_COLS)

                    dw[:, t] = dw_t.reshape(npx)

                srtm = srtm_h.read(window=win).astype(np.float32)
                srtm_bad = np.any(srtm == NODATA_I16, axis=0).reshape(npx)
                srtm = np.where(srtm == NODATA_I16, 0.0, srtm)
                srtm[1] /= SLOPE_SCALE
                srtm = srtm.reshape(2, npx).T
                x_raw[:, :, SRTM_RAW_IDXS] = srtm[:, None, :]
                if np.any(srtm_bad):
                    mask[np.ix_(np.flatnonzero(srtm_bad), np.arange(T), SRTM_COLS)] = 1

                rows = np.repeat(np.arange(r0, r1), W)
                cols = np.tile(np.arange(W), ch)
                xs, ys = _pixel_xy(ref_transform, rows, cols)
                er, ec = _rowcol_floor(era5_transform, xs, ys, era5_h, era5_w)
                x_raw[:, :, ERA5_RAW_IDXS] = era5[:, :, er, ec].transpose(2, 0, 1)

                lon, lat = to_wgs84.transform(xs, ys)
                latlon = np.stack([lat, lon], -1).astype(np.float32)

                mask[:, TARGET_T:, OPTICAL_COLS] = 1
                dw[:, TARGET_T:] = DW_MISSING

                x = NORMALIZE_FN(x_raw)
                x[:, TARGET_T:, OPTICAL_COLS] = 0.0

                months = np.broadcast_to(MONTHS_WINDOW, (npx, T)).copy()
                preds = np.full(npx, np.nan, dtype=np.float32)
                counts = token_counts(mask, dw, DW_MISSING)

                x_t = torch.from_numpy(x)
                dw_t = torch.from_numpy(dw)
                lat_t = torch.from_numpy(latlon)
                mask_t = torch.from_numpy(mask)
                month_t = torch.from_numpy(months)

                unique_counts = np.unique(counts)
                n_batches = 0
                groups = []
                for cv in unique_counts:
                    grp = np.where(counts == cv)[0]
                    groups.append(grp)
                    n_batches += (len(grp) + BATCH_SIZE - 1) // BATCH_SIZE
                print(
                    f"{tag}Chunk {cid + 1}/{TOTAL_CHUNKS}: inferring {npx} pixels "
                    f"in {n_batches} batches...",
                    flush=True,
                )
                b_i = 0
                t_last = time.perf_counter()
                for grp in groups:
                    for s in range(0, len(grp), BATCH_SIZE):
                        sel = grp[s:s + BATCH_SIZE]
                        sel_t = torch.from_numpy(sel)
                        recon, _ = model(
                            x=x_t.index_select(0, sel_t).to(device, non_blocking=True),
                            dynamic_world=dw_t.index_select(0, sel_t).to(device, non_blocking=True),
                            latlons=lat_t.index_select(0, sel_t).to(device, non_blocking=True),
                            mask=mask_t.index_select(0, sel_t).to(device, non_blocking=True),
                            month=month_t.index_select(0, sel_t).to(device, non_blocking=True),
                        )
                        preds[sel] = recon[:, TARGET_T, NDVI_IDX].cpu().numpy()
                        b_i += 1
                        now = time.perf_counter()
                        if b_i == 1 or b_i == n_batches or (now - t_last) >= 15:
                            print(
                                f"{tag}Chunk {cid + 1}/{TOTAL_CHUNKS}: "
                                f"batch {b_i}/{n_batches}",
                                flush=True,
                            )
                            t_last = now

                stale = np.full(npx, T * DEKAD_DAYS, dtype=np.int16)
                for t in range(T - 1):
                    has = s2_valid[:, t]
                    stale[has] = (T - 1 - t) * DEKAD_DAYS + s2_age[has, t]

                np.savez_compressed(
                    c_path,
                    ndvi=preds.reshape(ch, W).astype(np.float32),
                    stale=stale.reshape(ch, W),
                )
                n_done += 1
                dt = time.perf_counter() - t0
                remaining = len(assigned) - n_done
                eta = dt * remaining
                print(
                    f"{tag}Computed chunk {cid + 1}/{TOTAL_CHUNKS} in {dt:.1f}s"
                    f"  (~{eta / 60:.1f} min left on this worker)",
                    flush=True,
                )
    finally:
        for h in hs.values():
            h.close()
        srtm_h.close()

    print(
        f"{'w' + str(rank) + ' ' if n_workers > 1 else ''}"
        f"Worker done: {n_done} new chunks in {time.perf_counter() - t_all:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Process 3 chunks only")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto uses CUDA if present, otherwise CPU. MPS is opt-in (slower on torch 2.1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel chunk workers. Keep 1 on CPU: extra workers oversubscribe PyTorch and slow this model down.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--rows-per-chunk", type=int, default=50)
    parser.add_argument(
        "--delete-cache",
        action="store_true",
        default=False,
        help="If set, delete obsolete aoi_chunks_* cache folders. Default: keep them.",
    )
    args = parser.parse_args()

    device = _select_device(args.device)
    ncpu = os.cpu_count() or 1
    workers = max(1, args.workers)
    if device.type != "cpu" and workers > 1:
        print(f"{device} does not scale across workers; using --workers 1")
        workers = 1
    cpu_threads = max(1, ncpu // workers)

    manifest_path = _first_existing([
        os.path.join(DOWNLOADS_DIR, "dekad_manifest.json"),
        os.path.join(ARTIFACTS_DIR, "dekad_manifest.json"),
    ])
    weights_path = _first_existing([
        os.environ.get("WEIGHTS_PATH"),
        os.path.join(ARTIFACTS_DIR, "hyderabad_ndvi_presto.pt"),
        os.path.join(DOWNLOADS_DIR, "hyderabad_ndvi_presto.pt"),
        os.path.join(BASE_DIR, "hyderabad_ndvi_presto.pt"),
    ])

    if not manifest_path:
        raise FileNotFoundError(
            f"Missing dekad_manifest.json in {DOWNLOADS_DIR} or {ARTIFACTS_DIR}."
        )
    if not weights_path:
        raise FileNotFoundError(
            "Missing hyderabad_ndvi_presto.pt. Looked in "
            f"{ARTIFACTS_DIR} and {DOWNLOADS_DIR}."
        )

    with open(manifest_path, "r") as f:
        ARCHIVE = json.load(f)

    DEKADS = ARCHIVE["dekads"]
    N_DEKADS = ARCHIVE["n_dekads"]
    T = ARCHIVE["context_length"]
    REFERENCE_DEKAD = N_DEKADS - 1
    WINDOW = DEKADS[REFERENCE_DEKAD - T + 1: REFERENCE_DEKAD + 1]
    REF_LABEL = DEKADS[REFERENCE_DEKAD]["label"]

    active_cache_folder = f"aoi_chunks_{REF_LABEL}"
    if args.delete_cache:
        print("Checking cache directory for stale chunks...")
        for entry in os.listdir(CACHE_DIR):
            entry_path = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(entry_path) and entry.startswith("aoi_chunks_"):
                if entry != active_cache_folder:
                    try:
                        shutil.rmtree(entry_path)
                        print(f"Pruned obsolete cache folder: {entry}")
                    except Exception as e:
                        print(f"Warning: Could not remove {entry}: {e}")
    else:
        print("Keeping existing cache folders (--delete-cache not set).")

    CHUNKS_DIR = os.path.join(CACHE_DIR, active_cache_folder)
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    print(f"Chunk cache: {CHUNKS_DIR}")

    with rasterio.open(tif("S2", WINDOW[0]["label"])) as src:
        W, H = src.width, src.height
        ref_transform = src.transform
        ref_crs_wkt = src.crs.to_wkt() if src.crs else "EPSG:4326"

    TOTAL_CHUNKS = (H + args.rows_per_chunk - 1) // args.rows_per_chunk
    if args.test_mode:
        TOTAL_CHUNKS = min(TOTAL_CHUNKS, 3)
        print(f"Test mode: processing {TOTAL_CHUNKS} chunks.")

    already = sum(
        1 for i in range(TOTAL_CHUNKS)
        if os.path.exists(os.path.join(CHUNKS_DIR, f"chunk_{i:05d}.npz"))
    )
    print(f"Using weights: {weights_path}")
    print(
        f"Running inference on device: {device}  workers={workers}  "
        f"threads/worker={cpu_threads}  batch={args.batch_size}"
    )
    print(f"Grid {W}x{H} -> {TOTAL_CHUNKS} chunks  ({already} already cached)")

    cfg_base = {
        "n_workers": workers,
        "cpu_threads": cpu_threads,
        "device": str(device),
        "archive": ARCHIVE,
        "weights_path": weights_path,
        "width": W,
        "height": H,
        "transform": tuple(ref_transform)[:6],
        "crs_wkt": ref_crs_wkt,
        "rows_per_chunk": args.rows_per_chunk,
        "batch_size": args.batch_size,
        "total_chunks": TOTAL_CHUNKS,
        "chunks_dir": CHUNKS_DIR,
    }

    t0 = time.perf_counter()
    if workers == 1:
        cfg_base["rank"] = 0
        _run_worker(cfg_base)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(workers):
            cfg = dict(cfg_base, rank=rank)
            p = ctx.Process(target=_run_worker, args=(cfg,))
            p.start()
            procs.append(p)
        exitcodes = []
        for p in procs:
            p.join()
            exitcodes.append(p.exitcode)
        bad = [c for c in exitcodes if c not in (0, None)]
        if bad:
            raise SystemExit(f"Worker failed with exit codes {exitcodes}")

    print(f"\nChunking complete in {time.perf_counter() - t0:.1f}s. Saved to {CHUNKS_DIR}")
    from assemble_mosaic import assemble
    assemble(
        chunks_dir=CHUNKS_DIR,
        ref_label=REF_LABEL,
        rows_per_chunk=args.rows_per_chunk,
    )


if __name__ == "__main__":
    main()
