"""Stitch cached AOI chunks into GeoTIFFs under outputs/."""

import json
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ARTIFACTS_DIR = os.environ.get(
    "NDVI_ARCHIVE_DIR",
    os.path.join(BASE_DIR, "GHMC_Dekadal_v2"),
)
ROWS_PER_CHUNK = 50


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _load_archive():
    path = _first_existing([
        os.path.join(DOWNLOADS_DIR, "dekad_manifest.json"),
        os.path.join(ARTIFACTS_DIR, "dekad_manifest.json"),
    ])
    if not path:
        raise FileNotFoundError("Missing dekad_manifest.json in downloads/.")
    with open(path) as f:
        return json.load(f)


def _grid_from_s2(label):
    tif = os.path.join(DOWNLOADS_DIR, f"S2_{label}.tif")
    if not os.path.isfile(tif):
        raise FileNotFoundError(f"Need {tif} for CRS/transform.")
    with rasterio.open(tif) as src:
        return src.width, src.height, src.transform, src.crs


def assemble(chunks_dir=None, ref_label=None, rows_per_chunk=ROWS_PER_CHUNK):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    archive = _load_archive()
    dekads = archive["dekads"]
    ref_label = ref_label or dekads[-1]["label"]
    ref_end = dekads[-1]["end"]
    chunks_dir = chunks_dir or os.path.join(CACHE_DIR, f"aoi_chunks_{ref_label}")
    if not os.path.isdir(chunks_dir):
        raise FileNotFoundError(f"Chunk cache not found: {chunks_dir}")

    W, H, transform, crs = _grid_from_s2(dekads[0]["label"])
    total = (H + rows_per_chunk - 1) // rows_per_chunk
    missing = [
        i for i in range(total)
        if not os.path.isfile(os.path.join(chunks_dir, f"chunk_{i:05d}.npz"))
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chunk(s) missing in {chunks_dir} "
            f"(first missing: chunk_{missing[0]:05d}.npz)."
        )

    ndvi_path = os.path.join(OUTPUTS_DIR, f"ndvi_{ref_label}.tif")
    stale_path = os.path.join(OUTPUTS_DIR, f"stale_{ref_label}.tif")
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "compress": "LZW",
        "blockxsize": 256,
        "blockysize": 256,
    }
    ndvi_profile = dict(profile, dtype="float32", nodata=float("nan"))
    stale_profile = dict(profile, dtype="int16", nodata=-1)

    print(f"Assembling {total} chunks ({W}x{H}) from {chunks_dir}")
    with rasterio.open(ndvi_path, "w", **ndvi_profile) as ndvi_ds:
        with rasterio.open(stale_path, "w", **stale_profile) as stale_ds:
            for cid in range(total):
                r0 = cid * rows_per_chunk
                data = np.load(os.path.join(chunks_dir, f"chunk_{cid:05d}.npz"))
                ndvi = data["ndvi"]
                stale = data["stale"]
                ch = ndvi.shape[0]
                win = Window(0, r0, W, ch)
                ndvi_ds.write(ndvi.astype(np.float32), 1, window=win)
                stale_ds.write(stale.astype(np.int16), 1, window=win)
                if cid == 0 or cid + 1 == total or (cid + 1) % 20 == 0:
                    print(f"  wrote chunk {cid + 1}/{total}", flush=True)

    png_path = os.path.join(OUTPUTS_DIR, f"ndvi_{ref_label}.png")
    _write_preview(ndvi_path, png_path, title=f"NDVI {ref_label}  (end {ref_end})")

    print(f"Wrote {ndvi_path}")
    print(f"Wrote {stale_path}")
    print(f"Wrote {png_path}")
    return ndvi_path, stale_path


def _write_preview(tif_path, png_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(tif_path) as src:
        arr = src.read(1)
    step = max(1, max(arr.shape) // 2000)
    vis = arr[::step, ::step]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(vis, cmap="RdYlGn", vmin=0.0, vmax=0.8)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="NDVI")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())
