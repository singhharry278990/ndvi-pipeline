# ==============================================================================
# NOTEBOOK 3: Operational NDVI reconstruction for the reference date
# REQUIRES GPU
# ==============================================================================
#
# This is the production endpoint. It answers exactly one question:
#     "Optical is unusable right now. What is NDVI over GHMC today?"
#
# The reference date is the end of the newest complete dekad in the archive, so
# re-running Notebook 0 to append dekads and re-running this script is the
# whole operational loop. Nothing here is specific to a hardcoded month.
#
# Shared setup plus held-out prediction (Stages 1, 2, 6). Whole-AOI chunk
# inference lives in predict_chunks.py:
#   python predict_ndvi.py      # Stage 2 held-out tiles
#   python predict_chunks.py    # Stage 3–5 AOI reconstruction
#
# Every expensive stage is cached and resumable. After a disconnect, re-run:
# each stage reloads what exists and computes only what is missing.
#   Stage 1  Model            lazy, once per runtime
#   Stage 2  Validation       predictions cached; plots recomputed
#   Stage 6  Metadata         grid/CRS/transform serialized, never RAM-only
#
# Converted from Notebook_3_InferenceNEW_testmode_(3)_bulletproof.ipynb.
# Local paths only: GeoTIFFs in downloads/, weights in GHMC_Dekadal_v2,
# chunk cache in cache/aoi_chunks_<dekad>/.
#
# Dependencies:
#   torch numpy einops rasterio matplotlib pyproj
# Plus NASA Harvest Presto (cloned automatically if missing). Inference uses
# single_file_presto.py and the published band-normalization constants, so the
# rest of Presto's training stack (hurry.filesize, openmapflow, ee, webdataset)
# is not required.
#
# Environment:
#   NDVI_ARCHIVE_DIR   local archive folder (default: ./GHMC_Dekadal_v2)
#   NDVI_IMAGERY_DIR   GeoTIFF folder (default: ./downloads)
#   NDVI_CACHE_DIR     chunk/artifact cache (default: ./cache)
#   PRESTO_DIR         path to a nasaharvest/presto checkout
# ==============================================================================

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import matplotlib
if not IN_COLAB and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt


def _setup_presto():
    """Put nasaharvest/presto on sys.path. Clone it if it is not present."""
    env = os.environ.get("PRESTO_DIR")
    candidates = []
    if env:
        candidates.append(env)
    if IN_COLAB:
        candidates.append("/content/presto")
    candidates.append(os.path.join(_SCRIPT_DIR, "presto"))

    for dest in candidates:
        if os.path.isdir(dest) and os.path.isfile(
            os.path.join(dest, "single_file_presto.py")
        ):
            if dest not in sys.path:
                sys.path.insert(0, dest)
            print(f"Using Presto at {dest}")
            return dest

    dest = "/content/presto" if IN_COLAB else os.path.join(_SCRIPT_DIR, "presto")
    print(f"Cloning https://github.com/nasaharvest/presto.git into {dest} ...")
    subprocess.check_call(
        ["git", "clone", "-q", "https://github.com/nasaharvest/presto.git", dest]
    )
    sys.path.insert(0, dest)
    return dest


_setup_presto()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    import einops  # noqa: F401
except ImportError:
    raise ImportError(
        "einops is required (used by Presto). Install with:  pip install einops"
    ) from None

import single_file_presto as presto_pkg

# Band list and (shift, scale) are copied from nasaharvest/presto
# presto/dataops/pipelines/s1_s2_era5_srtm.py so we do not import that module.
# Importing `presto.dataops` pulls dataset.py, which needs hurry.filesize,
# openmapflow, Earth Engine, and webdataset -- none of which inference uses.
# `pip install hurry` is the wrong package; the real extra is hurry.filesize,
# and even that is only needed for Presto's training data pipeline.
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
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]
DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS
DYNAMIC_BANDS_SHIFT = S1_SHIFT_VALUES + S2_SHIFT_VALUES + ERA5_SHIFT_VALUES
DYNAMIC_BANDS_DIV = S1_DIV_VALUES + S2_DIV_VALUES + ERA5_DIV_VALUES
REMOVED_BANDS = ["B1", "B10"]
BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + STATIC_BANDS + ["NDVI"]
ADD_BY = (
    [DYNAMIC_BANDS_SHIFT[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + SRTM_SHIFT_VALUES
    + [0.0]
)
DIVIDE_BY = (
    [DYNAMIC_BANDS_DIV[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVED_BANDS]
    + SRTM_DIV_VALUES
    + [1.0]
)
NORMED_BANDS = [x for x in BANDS if x != "B9"]


def _calculate_ndvi(input_array):
    band_1, band_2 = "B8", "B4"
    num_dims = len(input_array.shape)
    if num_dims == 2:
        band_1_np = input_array[:, NORMED_BANDS.index(band_1)]
        band_2_np = input_array[:, NORMED_BANDS.index(band_2)]
    elif num_dims == 3:
        band_1_np = input_array[:, :, NORMED_BANDS.index(band_1)]
        band_2_np = input_array[:, :, NORMED_BANDS.index(band_2)]
    else:
        raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        if isinstance(band_1_np, np.ndarray):
            return np.where(
                (band_1_np + band_2_np) > 0,
                (band_1_np - band_2_np) / (band_1_np + band_2_np),
                0,
            )
        return torch.where(
            (band_1_np + band_2_np) > 0,
            (band_1_np - band_2_np) / (band_1_np + band_2_np),
            0,
        )


def NORMALIZE_FN(x):
    keep_indices = [idx for idx, val in enumerate(BANDS) if val != "B9"]
    if isinstance(x, np.ndarray):
        x = ((x + ADD_BY) / DIVIDE_BY).astype(np.float32)
    else:
        x = (x + torch.tensor(ADD_BY)) / torch.tensor(DIVIDE_BY)
    if len(x.shape) == 2:
        x = x[:, keep_indices]
        x[:, NORMED_BANDS.index("NDVI")] = _calculate_ndvi(x)
    else:
        x = x[:, :, keep_indices]
        x[:, :, NORMED_BANDS.index("NDVI")] = _calculate_ndvi(x)
    return x

ARTIFACTS_DIR = os.environ.get(
    "NDVI_ARCHIVE_DIR",
    os.path.join(_SCRIPT_DIR, "GHMC_Dekadal_v2"),
)
IMAGERY_DIR = os.environ.get(
    "NDVI_IMAGERY_DIR",
    os.path.join(_SCRIPT_DIR, "downloads"),
)
CACHE_DIR = os.environ.get(
    "NDVI_CACHE_DIR",
    os.path.join(_SCRIPT_DIR, "cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_DIR = os.path.join(ARTIFACTS_DIR, "presto_dataset")
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "hyderabad_ndvi_presto.pt")

# NDVI is a normalized difference of two bands that share ADD_BY=0 and the same
# DIVIDE_BY, so the scale cancels: normalized NDVI equals raw NDVI and Stage 2
# and Stage 5 are already on one scale.
_b4, _b8 = BANDS.index("B4"), BANDS.index("B8")
assert ADD_BY[_b4] == 0.0 and ADD_BY[_b8] == 0.0
assert DIVIDE_BY[_b4] == DIVIDE_BY[_b8]
print(f"Archive dir: {ARTIFACTS_DIR}")
print(f"Imagery dir: {IMAGERY_DIR}")
print(f"Chunk cache: {CACHE_DIR}")


def show_or_save(name):
    """Show in Colab; always write a PNG so a headless local run still has plots."""
    plt.tight_layout()
    out = os.path.join(CACHE_DIR, name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out}")
    if IN_COLAB:
        plt.show()
    else:
        plt.close()

# ------------------------------------------------------------------------------
# CELL 2: Cache layer -- atomic writes and stage guards
# ------------------------------------------------------------------------------

FORCE_RECOMPUTE = {
    "reference_metadata": False,
    "val_predictions": False,
    "aoi_inference": False,
    "final_raster": False,
    "target_truth": False,
}


def _atomic_write(path, write_fn):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_npy(path, arr):
    _atomic_write(path, lambda f: np.save(f, arr, allow_pickle=False))


def save_npz(path, **a):
    _atomic_write(path, lambda f: np.savez_compressed(f, **a))


def save_json(path, obj):
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def file_sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def cached(key, path, compute, load, save, label=None):
    label = label or key
    if os.path.exists(path) and not FORCE_RECOMPUTE.get(key, False):
        print(f"✓ {label}: loaded from cache  ({os.path.basename(path)}, "
              f"{human(os.path.getsize(path))})")
        return load(path)
    print(f"·  {label}: "
          + ("FORCE_RECOMPUTE set -- recomputing" if os.path.exists(path)
             else "not cached -- computing"))
    obj = compute()
    save(path, obj)
    print(f"✓ {label}: computed and saved  ({human(os.path.getsize(path))})")
    return obj


print("Cache layer ready.")


# ------------------------------------------------------------------------------
# CELL 3: STAGE 6 -- constants, reference date, serialized grid metadata
# ------------------------------------------------------------------------------

from collections import OrderedDict

ARCHIVE_PATH = next(
    (p for p in (
        os.path.join(IMAGERY_DIR, "dekad_manifest.json"),
        os.path.join(ARTIFACTS_DIR, "dekad_manifest.json"),
    ) if os.path.isfile(p)),
    None,
)
assert ARCHIVE_PATH, (
    f"No dekadal archive manifest in {IMAGERY_DIR} or {ARTIFACTS_DIR}. "
    "Run the download script first."
)
ARCHIVE = load_json(ARCHIVE_PATH)

DEKADS = ARCHIVE["dekads"]
N_DEKADS = ARCHIVE["n_dekads"]
DEKAD_DAYS = ARCHIVE["dekad_days"]
NODATA_I16 = ARCHIVE["nodata_i16"]
S1_SCALE = ARCHIVE["s1_scale"]
SLOPE_SCALE = ARCHIVE["slope_scale"]
ARCHIVE_S2_BANDS = ARCHIVE["s2_bands"]

# Same column layout Notebook 1 writes into presto_dataset/manifest.json.
# Derived here so Stage 3 can run without that dataset (Notebook 1) locally.
N_IDX = {b: i for i, b in enumerate(NORMED_BANDS)}
_derived = {
    "num_timesteps": int(ARCHIVE["context_length"]),
    "target_slot": int(ARCHIVE["context_length"]) - 1,
    "ndvi_idx": int(N_IDX["NDVI"]),
    "optical_cols": sorted(
        [N_IDX[b] for b in ARCHIVE_S2_BANDS] + [N_IDX["NDVI"]]
    ),
    "s1_cols": [N_IDX["VV"], N_IDX["VH"]],
    "srtm_cols": [N_IDX["elevation"], N_IDX["slope"]],
    "dw_missing_class": int(ARCHIVE["dw_missing_class"]),
}

CFG_PATH = f"{ARTIFACTS_DIR}/finetune_config.json"
DS_MANIFEST_PATH = f"{DATASET_DIR}/manifest.json"
if os.path.isfile(CFG_PATH):
    cfg = load_json(CFG_PATH)
    print(f"✓ Using {os.path.basename(CFG_PATH)}")
else:
    cfg = dict(_derived)
    print("·  finetune_config.json not found (Notebook 2 has not been run "
          "locally). Deriving band layout from dekad_manifest.json.")

if os.path.isfile(DS_MANIFEST_PATH):
    manifest = load_json(DS_MANIFEST_PATH)
    print(f"✓ Using {DS_MANIFEST_PATH}")
else:
    manifest = dict(_derived)
    print("·  presto_dataset/manifest.json not found (Notebook 1 has not been "
          "run locally). Stage 2 held-out validation will be skipped.")

T = cfg.get("num_timesteps", _derived["num_timesteps"])
TARGET_T = cfg.get("target_slot", _derived["target_slot"])
NDVI_IDX = cfg.get("ndvi_idx", _derived["ndvi_idx"])
OPTICAL_COLS = cfg.get("optical_cols", _derived["optical_cols"])
S1_COLS = cfg.get("s1_cols", _derived["s1_cols"])
SRTM_COLS = manifest.get("srtm_cols", _derived["srtm_cols"])
DW_MISSING = cfg.get("dw_missing_class", _derived["dw_missing_class"])

# ---- the reference date -------------------------------------------------------
# Default: the newest dekad in the archive, i.e. "now". Pin it to an older index
# to reproduce a past run.
REFERENCE_DEKAD = N_DEKADS - 1
assert REFERENCE_DEKAD >= T - 1, "Not enough history for a full window."
WINDOW = DEKADS[REFERENCE_DEKAD - T + 1: REFERENCE_DEKAD + 1]
REF_LABEL = DEKADS[REFERENCE_DEKAD]["label"]
REF_DATE = DEKADS[REFERENCE_DEKAD]["end"]
MONTHS_WINDOW = np.array([d["month_index"] for d in WINDOW], dtype=np.int64)

BANDS_GROUPS_IDX = OrderedDict([
    ("S1", [0, 1]), ("S2_RGB", [2, 3, 4]), ("S2_Red_Edge", [5, 6, 7]),
    ("S2_NIR_10m", [8]), ("S2_NIR_20m", [9]), ("S2_SWIR", [10, 11]),
    ("ERA5", [12, 13]), ("SRTM", [14, 15]), ("NDVI", [16]),
])
RAW_IDX = {b: i for i, b in enumerate(BANDS)}
S2_RAW_IDXS = [RAW_IDX[b] for b in ARCHIVE_S2_BANDS]
S1_RAW_IDXS = [RAW_IDX['VV'], RAW_IDX['VH']]
ERA5_RAW_IDXS = [RAW_IDX['temperature_2m'], RAW_IDX['total_precipitation']]
SRTM_RAW_IDXS = [RAW_IDX['elevation'], RAW_IDX['slope']]
B4_POS = ARCHIVE_S2_BANDS.index('B4')
B8_POS = ARCHIVE_S2_BANDS.index('B8')


def tif(prefix, label=None):
    name = f"{prefix}_{label}.tif" if label else f"{prefix}.tif"
    for folder in (IMAGERY_DIR, ARTIFACTS_DIR):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return os.path.join(IMAGERY_DIR, name)


REF_META_PATH = f"{CACHE_DIR}/reference_metadata.json"


def _compute_ref_meta():
    path = tif('S2', WINDOW[0]['label'])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Need {path} to read the AOI grid. Wait until the Notebook 0 "
            "GeoTIFF download has that file, then re-run."
        )
    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height,
                "transform": list(src.transform)[:6], "crs": src.crs.to_string(),
                "driver": src.driver, "reference_dekad": REF_LABEL,
                "reference_date": REF_DATE,
                "window": [d["label"] for d in WINDOW]}


ref = cached("reference_metadata", REF_META_PATH, _compute_ref_meta,
             load_json, save_json, "Stage 6  reference raster metadata")
if ref.get("reference_dekad") != REF_LABEL:
    # Expected every time the archive has advanced since this cache was
    # written. This file only stores grid transform/CRS/dims -- cheap to
    # recompute (one rasterio.open) -- so a mismatch is not an error state,
    # just a signal to refresh it.
    print(f"·  Stage 6 reference raster metadata: cached copy is for dekad "
          f"{ref.get('reference_dekad')}, this run targets {REF_LABEL} -- "
          f"recomputing.")
    ref = _compute_ref_meta()
    save_json(REF_META_PATH, ref)

ref_transform = rasterio.Affine(*ref["transform"])
ref_crs = rasterio.crs.CRS.from_string(ref["crs"])
H, W = ref["height"], ref["width"]
ref_meta = {"driver": ref.get("driver", "GTiff"), "width": W, "height": H,
            "count": 1, "dtype": "float32", "crs": ref_crs,
            "transform": ref_transform, "nodata": float("nan")}

print(f"\nReference date: {REF_DATE}  (dekad {REFERENCE_DEKAD}, label {REF_LABEL})")
print(f"Context window: {WINDOW[0]['start']} .. {WINDOW[-1]['end']} "
      f"({T} slots x {DEKAD_DAYS}d = {T * DEKAD_DAYS} days)")
print(f"AOI grid: {W} x {H} ({H * W:,} pixels)")


# ------------------------------------------------------------------------------
# CELL 4: STAGE 1 -- model, loaded lazily and at most once per runtime
# ------------------------------------------------------------------------------

_MODEL_CACHE = {}


def get_model():
    if "model" in _MODEL_CACHE:
        print("✓ Stage 1  model already loaded in this runtime -- reusing.")
        return _MODEL_CACHE["model"]
    print("·  Stage 1  loading fine-tuned weights ...")
    m = presto_pkg.Presto.construct()
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    m = m.to(device).eval()
    _MODEL_CACHE["model"] = m
    print(f"✓ Stage 1  fine-tuned model loaded onto {device}.")
    return m


if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"No fine-tuned weights at {WEIGHTS_PATH}.\n"
        "Put hyderabad_ndvi_presto.pt in GHMC_Dekadal_v2 (Notebook 2 output)."
    )
WEIGHTS_SHA1 = file_sha1(WEIGHTS_PATH)
print(f"Weights present: {human(os.path.getsize(WEIGHTS_PATH))}, "
      f"sha1 {WEIGHTS_SHA1[:12]} (loaded on first use)")


# ==============================================================================
# STAGE 2: Held-out validation on the exact deployment task
# ==============================================================================
# Same tiles Notebook 2 held out, same masking, k=0 so the staleness is the
# staleness the data really has. Predictions are cached; a re-run does no
# inference at all.

VAL_PRED_PATH = f"{CACHE_DIR}/val_predictions.npz"
STAGE2_INPUTS = [
    f"{DATASET_DIR}/x.pt",
    f"{DATASET_DIR}/mask.pt",
    f"{DATASET_DIR}/dynamic_world.pt",
    f"{DATASET_DIR}/latlons.pt",
    f"{DATASET_DIR}/months.pt",
    f"{DATASET_DIR}/staleness.pt",
    f"{ARTIFACTS_DIR}/split_indices.pt",
]


def metrics(p, t):
    e = p - t
    ss = float(np.sum((t - t.mean()) ** 2))
    return {"n": len(t), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": 1 - float(np.sum(e ** 2)) / ss if ss > 0 else float("nan"),
            "std_true": float(t.std()), "std_pred": float(p.std()),
            "std_ratio": float(p.std() / t.std()) if t.std() > 0 else float("nan")}


def run_stage2():
    """Predict held-out tiles. Returns (pred, true, stale) or (None, None, None)."""
    skip = any(not os.path.isfile(p) for p in STAGE2_INPUTS)

    if os.path.exists(VAL_PRED_PATH) and not FORCE_RECOMPUTE["val_predictions"]:
        _d = np.load(VAL_PRED_PATH)
        v_pred, v_true, v_stale = _d["pred"], _d["true"], _d["stale"]
        print(f"✓ Stage 2  validation predictions loaded from cache "
              f"({human(os.path.getsize(VAL_PRED_PATH))}). Inference skipped.")
        print(f"   {len(v_true):,} held-out samples")
        return v_pred, v_true, v_stale

    if skip:
        missing = [p for p in STAGE2_INPUTS if not os.path.isfile(p)]
        print("⚠ Stage 2 skipped -- Notebook 1/2 dataset is not local.")
        print("   Missing:")
        for p in missing:
            print(f"      {os.path.relpath(p, ARTIFACTS_DIR)}")
        print("   Stage 2 is optional held-out validation. Continuing to whole-AOI NDVI.")
        return None, None, None

    print("·  Stage 2  no cached validation predictions -- running inference")
    model = get_model()

    x_all = torch.load(f"{DATASET_DIR}/x.pt")
    mask_all = torch.load(f"{DATASET_DIR}/mask.pt")
    dw_all = torch.load(f"{DATASET_DIR}/dynamic_world.pt")
    latlon_all = torch.load(f"{DATASET_DIR}/latlons.pt")
    months_all = torch.load(f"{DATASET_DIR}/months.pt")
    stale_all = torch.load(f"{DATASET_DIR}/staleness.pt")
    val_idx = torch.load(f"{ARTIFACTS_DIR}/split_indices.pt")["val_idx"]

    def token_mask(mask, dw):
        toks = []
        for name, idxs in BANDS_GROUPS_IDX.items():
            g = mask[:, :, idxs].max(dim=-1).values > 0
            toks.append(g[:, :1] if name == "SRTM" else g)
        toks.append(dw == DW_MISSING)
        return torch.cat(toks, dim=1)

    m_all = mask_all[val_idx].clone()
    d_all = dw_all[val_idx].clone()
    m_all[:, TARGET_T:, OPTICAL_COLS] = 1
    d_all[:, TARGET_T:] = DW_MISSING
    counts = token_mask(m_all, d_all).sum(dim=1).numpy()

    preds, trues = [], []
    order = []
    with torch.no_grad():
        for c in np.unique(counts):
            grp = np.where(counts == c)[0]
            for s in range(0, len(grp), 512):
                sel = grp[s:s + 512]
                gi = val_idx[sel]
                x = x_all[gi].clone()
                y = x[:, TARGET_T, NDVI_IDX].clone()
                x[:, TARGET_T:, OPTICAL_COLS] = 0.0
                recon, _ = model(
                    x=x.to(device), dynamic_world=d_all[sel].to(device),
                    latlons=latlon_all[gi].to(device), mask=m_all[sel].to(device),
                    month=months_all[gi].to(device))
                preds.append(recon[:, TARGET_T, NDVI_IDX].cpu().numpy())
                trues.append(y.numpy())
                order.append(sel)

    order = np.concatenate(order)
    v_pred, v_true = np.concatenate(preds), np.concatenate(trues)
    v_stale = stale_all[val_idx].numpy()[order]
    save_npz(VAL_PRED_PATH, pred=v_pred, true=v_true, stale=v_stale)
    print(f"✓ Stage 2  computed and cached ({human(os.path.getsize(VAL_PRED_PATH))}).")
    print(f"   {len(v_true):,} held-out samples")
    return v_pred, v_true, v_stale


def run_stage2_report(v_pred, v_true, v_stale):
    """Print staleness-stratified metrics and save the Stage 2 figure."""
    if v_pred is None:
        print("⚠ Stage 2 metrics skipped (no held-out predictions).")
        return

    ov = metrics(v_pred, v_true)
    print("--- Stage 2: held-out tiles, reference-date NDVI ---")
    print(f"{'staleness of last clear look':<30}{'n':>9}{'MAE':>9}{'RMSE':>9}"
          f"{'R2':>9}{'std(true)':>11}{'ratio':>8}")
    print(f"{'ALL':<30}{ov['n']:>9,}{ov['MAE']:>9.4f}{ov['RMSE']:>9.4f}"
          f"{ov['R2']:>9.4f}{ov['std_true']:>11.4f}{ov['std_ratio']:>8.3f}")

    edges = [0, 10, 20, 30, 60, 120, 10_000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v_stale >= lo) & (v_stale < hi)
        if sel.sum() < 50:
            continue
        m = metrics(v_pred[sel], v_true[sel])
        lbl = f"{lo}-{hi if hi < 10_000 else '240+'} days"
        print(f"{lbl:<30}{m['n']:>9,}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}"
              f"{m['R2']:>9.4f}{m['std_true']:>11.4f}{m['std_ratio']:>8.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(v_true, v_pred, s=3, alpha=0.25)
    lims = [min(v_true.min(), v_pred.min()), max(v_true.max(), v_pred.max())]
    ax[0].plot(lims, lims, "r--")
    ax[0].set_xlabel("True NDVI"), ax[0].set_ylabel("Reconstructed NDVI")
    ax[0].set_title(f"R2={ov['R2']:.3f}  RMSE={ov['RMSE']:.4f}  "
                    f"std ratio={ov['std_ratio']:.3f}")
    bs, ms = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v_stale >= lo) & (v_stale < hi)
        if sel.sum() >= 50:
            bs.append(f"{lo}-{hi if hi < 10_000 else '240+'}")
            ms.append(float(np.mean(np.abs(v_pred[sel] - v_true[sel]))))
    ax[1].bar(bs, ms)
    ax[1].set_xlabel("days since last clear optical look"), ax[1].set_ylabel("MAE")
    ax[1].set_title("Accuracy decays with staleness")
    show_or_save("stage2_validation.png")


def run_aoi_chunks():
    """Operational NDVI: reconstruct the whole AOI via predict_chunks.py."""
    script = os.path.join(_SCRIPT_DIR, "predict_chunks.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Missing {script}")
    cmd = [sys.executable, script]
    if os.environ.get("TEST_MODE", "0") == "1" or "--test-mode" in sys.argv:
        cmd.append("--test-mode")
    print("\n--- Whole-AOI NDVI reconstruction ---")
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc:
        raise SystemExit(rc)


def main():
    v_pred, v_true, v_stale = run_stage2()
    run_stage2_report(v_pred, v_true, v_stale)
    if os.environ.get("SKIP_CHUNKS", "0") == "1":
        print("SKIP_CHUNKS=1 -- not running whole-AOI reconstruction.")
        return
    run_aoi_chunks()


if __name__ == "__main__":
    main()


