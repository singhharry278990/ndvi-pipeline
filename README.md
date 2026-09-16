# GHMC Dekadal NDVI Reconstruction Pipeline

This repository reconstructs **NDVI (Normalized Difference Vegetation Index)** over the Greater Hyderabad Municipal Corporation (GHMC) when optical satellite imagery is cloudy or otherwise unusable.

The production question is a single one:

> Optical is unusable right now. What is NDVI over GHMC today?

The answer is a georeferenced GeoTIFF of reconstructed NDVI at 10 m resolution, plus a staleness map that says how many days have passed since each pixel last had a clear Sentinel-2 look.

The model is a fine-tuned [NASA Harvest Presto](https://github.com/nasaharvest/presto) transformer. It never sees current optical bands at the target date. It infers NDVI from:

- a **240-day history** of Sentinel-2 (when it was clear)
- current and historical **Sentinel-1 radar** (cloud-penetrating)
- **ERA5** temperature and precipitation
- **SRTM** elevation and slope
- **Dynamic World** land-cover class
- pixel **latitude / longitude** and calendar **month**

---

## How the pieces fit together

```
                    Google Earth Engine
                    (S2, S1, ERA5, DW, SRTM)
                              │
                              ▼
                 download_predicted_files.py
                 rolling 24-dekad GeoTIFF archive
                              │
                              ▼
                       downloads/*.tif
                  downloads/dekad_manifest.json
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
   predict_ndvi.py                      GHMC_Dekadal_v2/
   optional held-out check              hyderabad_ndvi_presto.pt
           │                            finetune_config.json
           ▼
   predict_chunks.py
   hide optical at the latest dekad
   run Presto on 50-row tiles
           │
           ▼
   cache/aoi_chunks_<label>/chunk_*.npz
           │
           ▼
   assemble_mosaic.py
           │
           ▼
   outputs/ndvi_<label>.tif
   outputs/stale_<label>.tif
   outputs/ndvi_<label>.png
```

Operational loop on a new dekad:

1. `python download_predicted_files.py` — pull the newest 10-day composites, drop files older than the 24-dekad window.
2. `python predict_ndvi.py` — validate if the held-out dataset is present, then reconstruct the whole AOI.
3. Read `outputs/ndvi_<dekad>.tif`.

Re-runs are resumable. Cached chunks are reused; only missing chunks are inferred.

---

## Area, grid, and time

| Setting | Value |
| --- | --- |
| AOI | GHMC bounding box (~78.23–78.65 E, 17.20–17.58 N) |
| CRS | EPSG:32644 (UTM zone 44N) |
| Pixel size | 10 m |
| Grid | 4531 × 4212 ≈ 19.1 million pixels |
| Time unit | **dekad** = 10 days |
| Context | 24 dekads = 240 days |
| Target | last dekad in the archive (the newest complete 10-day window) |
| ERA5 resolution | 1000 m, sampled onto the 10 m grid |

A dekad is labelled by its **start date**. Example from the current archive:

| Field | Example |
| --- | --- |
| Label | `2026-09-01` |
| Window | `2026-09-01` .. `2026-09-11` |
| Context used | `2026-01-14` .. `2026-09-11` |

`download_predicted_files.py` keeps only those 24 dekads on disk. Older GeoTIFFs are pruned. SRTM is static and is never pruned.

The training archive in `GHMC_Dekadal_v2/dekad_manifest.json` can be longer (60 dekads in this checkout). Inference only needs the latest 24.

---

## Inputs per dekad

For every 10-day slot `YYYY-MM-DD`, Earth Engine composites are written under `downloads/`:

| File | Source | Bands | Scale | Role |
| --- | --- | --- | --- | --- |
| `S2_<label>.tif` | Sentinel-2 SR Harmonized | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 m, int16 reflectance × 10000 | Optical history. **Hidden at the target dekad.** |
| `S2OBS_<label>.tif` | derived from S2 | `n_obs`, `age_days` | 10 m, uint8 | Clear-pixel count and age of the newest clear observation inside the dekad |
| `S1_<label>.tif` | Sentinel-1 GRD IW | VV, VH | 10 m, int16 dB × 100 | Radar. Kept at every timestep, including the target. |
| `S1OBS_<label>.tif` | derived from S1 | `n_obs` | 10 m, uint8 | Radar observation count |
| `DW_<label>.tif` | Dynamic World | land-cover class 0–8, missing = 9 | 10 m, uint8 | Land-cover token. Target slot is forced to missing. |
| `ERA5_<label>.tif` | ERA5-Land daily | 2 m temperature, precipitation (monthly-equivalent) | 1000 m, float32 | Weather. Kept at every timestep. |
| `SRTM_Static.tif` | SRTM GL1 | elevation, slope × 100 | 10 m, int16 | Terrain. Repeated across all 24 slots. |

Sentinel-2 cloud masking uses SCL bad classes `{1, 3, 8, 9, 10, 11}` and Cloud Score+ `cs_cdf ≥ 0.60`. Remaining pixels are a **median composite**. If a pixel has no clear look in that dekad, it is nodata (`-32768`) and the model mask marks optical bands as missing.

ERA5 for the previous dekad is re-downloaded once (“late fill”) so the precipitation window is complete after ECMWF finishes publishing.

---

## What the model actually predicts

NDVI from optical is:

```
NDVI = (NIR − Red) / (NIR + Red) = (B8 − B4) / (B8 + B4)
```

At the **target dekad** the pipeline zeros every optical column (S2 bands + NDVI) and sets Dynamic World to class 9 (missing). The transformer must reconstruct NDVI from radar, weather, terrain, location, season, and the earlier optical history.

That is the deployment task the weights were fine-tuned for (`GHMC_Dekadal_v2/finetune_config.json`):

- 24 timesteps, target slot = 23 (the last dekad)
- encoder + decoder fine-tune
- tile-disjoint 25% validation
- held-out RMSE ≈ **0.090**, MAE ≈ **0.054**, R² ≈ **0.88**

Months are passed as a 2-D `[batch, T]` tensor. Presto’s default 1-D month path would increment one month per timestep, which is wrong at a 10-day cadence.

### Band vector after Presto normalization

Each pixel × timestep is a 17-band vector (`B1` and `B10` dropped; `B9` dropped after shift/scale):

| Index | Group | Bands |
| --- | --- | --- |
| 0–1 | Sentinel-1 | VV, VH |
| 2–4 | S2 RGB | B2, B3, B4 |
| 5–7 | S2 red edge | B5, B6, B7 |
| 8 | S2 NIR 10 m | B8 |
| 9 | S2 NIR 20 m | B8A |
| 10–11 | S2 SWIR | B11, B12 |
| 12–13 | ERA5 | temperature_2m, total_precipitation |
| 14–15 | SRTM | elevation, slope |
| 16 | NDVI | computed from normalized B8 and B4 |

Normalization constants match NASA Harvest Presto (`shift` / `scale` per sensor). After that, target-dekad optical values are overwritten with 0 so they cannot leak into the prediction.

---

## Repository layout

```
ndviPipeline/
├── download_predicted_files.py   # Notebook 0: GEE → local GeoTIFFs
├── predict_ndvi.py               # Notebook 3 setup + optional Stage 2 validation
├── predict_chunks.py             # whole-AOI inference (50-row chunks)
├── assemble_mosaic.py            # stitch chunks → GeoTIFF + PNG preview
├── requirements.txt
├── .env.example                  # Earth Engine project + service-account path
├── downloads/                    # rolling imagery archive (not committed)
│   ├── dekad_manifest.json
│   ├── S2_*.tif, S1_*.tif, ERA5_*.tif, DW_*.tif, S2OBS_*.tif, S1OBS_*.tif
│   └── SRTM_Static.tif
├── GHMC_Dekadal_v2/              # trained artifacts
│   ├── hyderabad_ndvi_presto.pt  # fine-tuned weights (required)
│   ├── finetune_config.json
│   └── dekad_manifest.json       # longer training archive metadata
├── cache/                        # resumable chunk + metadata cache
│   └── aoi_chunks_<label>/
├── outputs/                      # final rasters
└── presto/                       # NASA Harvest Presto checkout (single_file_presto.py)
```

Training notebooks 1 and 2 (pixel dataset + fine-tune) are **not** required to run inference. If `presto_dataset/` and `split_indices.pt` are missing, Stage 2 validation is skipped and the pipeline still builds the AOI mosaic.

---

## Scripts

### 1. `download_predicted_files.py`

Builds a 24-dekad rolling archive from Earth Engine and writes GeoTIFFs straight to `downloads/` (no Google Drive).

- Authenticates with a **service account** (`EE_PROJECT` + `GOOGLE_APPLICATION_CREDENTIALS` in `.env`). User OAuth is not used; this project bills the service-account project.
- Advances `archive_end` by whole dekads when enough days have passed (`ADVANCE_ARCHIVE = True`).
- Skips files that already exist.
- Deletes GeoTIFFs whose dekad label is no longer in the window.
- Retries each download up to 3 times.

Copy `.env.example` to `.env` and point it at the JSON key:

```bash
cp .env.example .env
# EE_PROJECT=ee-earth-engine-503704
# GOOGLE_APPLICATION_CREDENTIALS=your-service-account.json
python download_predicted_files.py
```

### 2. `predict_ndvi.py`

Entry point for a full run.

| Stage | What it does | Cache |
| --- | --- | --- |
| Setup | Resolve Presto, device, archive, band layout | — |
| Stage 6 | Read CRS / transform / width / height from an S2 raster | `cache/reference_metadata.json` |
| Stage 1 | Load `hyderabad_ndvi_presto.pt` (lazy, first use) | in-memory |
| Stage 2 | Held-out tile metrics + scatter/MAE plots | `cache/val_predictions.npz`, `cache/stage2_validation.png` |
| Chunks | Delegates to `predict_chunks.py` | `cache/aoi_chunks_<label>/` |

Stage 2 is optional. Without `presto_dataset/{x,mask,dynamic_world,latlons,months,staleness}.pt` and `split_indices.pt`, it prints a warning and continues.

```bash
python predict_ndvi.py
```

Useful environment flags:

| Variable | Effect |
| --- | --- |
| `SKIP_CHUNKS=1` | Stage 2 only, no AOI inference |
| `TEST_MODE=1` | Same as `--test-mode` on the chunk script (3 chunks) |
| `NDVI_ARCHIVE_DIR` | Weights / config folder (default `GHMC_Dekadal_v2`) |
| `NDVI_IMAGERY_DIR` | GeoTIFF folder (default `downloads`) |
| `NDVI_CACHE_DIR` | Chunk cache (default `cache`) |
| `PRESTO_DIR` | Path to a nasaharvest/presto checkout |

### 3. `predict_chunks.py`

Whole-AOI reconstruction.

1. Split the 4212-row grid into strips of **50 rows** (~85 chunks).
2. For each strip, read the 24-dekad stack (S2, S1, DW, ERA5, SRTM).
3. Build Presto tensors, **mask optical at the last slot**, run the model in batches of 4096 pixels grouped by token-mask count (variable-length masking is more efficient when the mask cardinality matches).
4. Write `chunk_XXXXX.npz` with `ndvi` and `stale`.
5. Call `assemble_mosaic.assemble()` when every chunk exists.

Staleness (days) for a pixel:

```
days since last clear S2 = (target_slot − last_clear_slot) × 10 + age_days_in_that_dekad
```

If there is no clear look in the 240-day window, staleness is `24 × 10 = 240`.

Already-computed chunks are skipped. Cache folders for **older** reference dekads are deleted so disk does not grow without bound.

```bash
python predict_chunks.py
python predict_chunks.py --test-mode
python predict_chunks.py --device cpu --batch-size 4096 --rows-per-chunk 50
python predict_chunks.py --device cuda
```

`--workers` defaults to 1. Extra CPU workers oversubscribe PyTorch on this model and usually slow it down. CUDA / MPS stay at 1 worker. Device `auto` prefers CUDA, then CPU. MPS is opt-in (`--device mps`); boolean masking is slower on torch 2.1 MPS than on Apple CPU.

### 4. `assemble_mosaic.py`

Stitches `cache/aoi_chunks_<label>/` into:

| File | Type | Meaning |
| --- | --- | --- |
| `outputs/ndvi_<label>.tif` | float32 GeoTIFF, LZW, tiled 256 | Reconstructed NDVI. Nodata = NaN |
| `outputs/stale_<label>.tif` | int16 GeoTIFF | Days since last clear optical look. Nodata = −1 |
| `outputs/ndvi_<label>.png` | preview | RdYlGn, NDVI 0–0.8 |

If `predict_chunks.py` already assembled the mosaic, you only need this script after a crash between chunking and stitching:

```bash
python assemble_mosaic.py
```

---

## Setup

Python **3.9–3.12**. Inference does **not** need Presto’s training extras (`hurry.filesize`, `openmapflow`, Earth Engine inside the model package). It uses `presto/single_file_presto.py` only.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU (example CUDA 12.1):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

On Ubuntu, install GDAL before `rasterio` if wheels fail:

```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev python3-dev build-essential
```

Place fine-tuned weights at:

```
GHMC_Dekadal_v2/hyderabad_ndvi_presto.pt
```

If `presto/` is missing, `predict_ndvi.py` clones https://github.com/nasaharvest/presto.git next to the scripts.

---

## Caching and how to force a recompute

| Artifact | Path | Invalidated when |
| --- | --- | --- |
| Grid metadata | `cache/reference_metadata.json` | Reference dekad label changes |
| Validation preds | `cache/val_predictions.npz` | `FORCE_RECOMPUTE["val_predictions"]` in `predict_ndvi.py` |
| AOI chunks | `cache/aoi_chunks_<label>/` | New reference label (old folders deleted) |
| Final rasters | `outputs/` | Re-run assemble |

Writes use a temp file + `os.replace` so a crash does not leave a truncated `.npy` / `.json`.

To rebuild chunks for the current dekad, delete that folder and re-run:

```bash
rm -rf cache/aoi_chunks_2026-09-01
python predict_chunks.py
```

---

## Interpreting outputs

- **NDVI** is typically −1 to 1. Vegetation over GHMC is mostly 0–0.8. Water and bare soil sit lower. The PNG stretch is 0–0.8 for display only; the GeoTIFF is the full predicted value.
- **Stale** is an uncertainty proxy, not a confidence interval. Fine-tune metrics degrade as days-since-clear-optical grow. Treat pixels with staleness of many dekads as less trustworthy, especially through monsoon cloud.
- The model is **not** filling true optical NDVI on clear days in the output raster. Every pixel at the target dekad is a reconstruction with optical hidden, so the map is spatially consistent with the cloudy-day product.

---

## Training lineage (Notebooks 1–2)

Those notebooks are not in this repo. Their outputs are:

| Artifact | Produced by | Used for |
| --- | --- | --- |
| `presto_dataset/*.pt` | Notebook 1 | Stage 2 held-out check |
| `split_indices.pt` | Notebook 2 | Same |
| `hyderabad_ndvi_presto.pt` | Notebook 2 | All inference |
| `finetune_config.json` | Notebook 2 | Band indices, `target_slot`, DW missing class |

Fine-tune recipe (from `finetune_config.json`): 12 epochs, batch 256, lr `3e-4`, weight decay 0.05, NDVI loss weight 1.0, reflectance reconstruction weight 0.3, random extra optical-gap augmentation `k = 0..5` dekads.

---

## Environment variables

| Name | Default | Used by |
| --- | --- | --- |
| `EE_PROJECT` | — | download script |
| `GOOGLE_APPLICATION_CREDENTIALS` | — | download script |
| `NDVI_ARCHIVE_DIR` | `./GHMC_Dekadal_v2` | predict / assemble |
| `NDVI_ARTIFACTS_DIR` | same as archive dir | `predict_chunks.py` |
| `NDVI_IMAGERY_DIR` | `./downloads` | `predict_ndvi.py` |
| `NDVI_CACHE_DIR` | `./cache` | predict / chunks |
| `WEIGHTS_PATH` | `GHMC_Dekadal_v2/hyderabad_ndvi_presto.pt` | `predict_chunks.py` |
| `PRESTO_DIR` | `./presto` | `predict_ndvi.py` |
| `SKIP_CHUNKS` | `0` | `predict_ndvi.py` |
| `TEST_MODE` | `0` | `predict_ndvi.py` |

---

## Dependencies (inference)

```
torch>=2.0,<2.6
numpy>=1.23,<2.0
einops>=0.6
rasterio>=1.3
matplotlib>=3.6
pyproj>=3.4
```

Download extras: `earthengine-api`, `geemap`, `geedim`.

---

## License

This project’s wrapper code is MIT (see `LICENSE`). Presto under `presto/` has its own license from NASA Harvest. Earth Engine and Copernicus data remain subject to their provider terms.
