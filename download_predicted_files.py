import json
import os
import re
import time
from collections import defaultdict
from datetime import date, timedelta
import ee

# ------------------------------------------------------------------ SETUP
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Output directly to the local downloads/ folder
DOWNLOADS_DIR = os.path.join(_SCRIPT_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _load_dotenv(path):
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    return True


_env_path = os.path.join(_SCRIPT_DIR, ".env")
_load_dotenv(_env_path)

# Service-account JSON path and project come from .env
# (EE_PROJECT, GOOGLE_APPLICATION_CREDENTIALS). User OAuth cannot bill
# this project -- that is what produced USER_PROJECT_DENIED / 403.
GEE_PROJECT = os.environ.get("EE_PROJECT", "").strip()
EE_KEY_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
if not GEE_PROJECT or not EE_KEY_FILE:
    raise RuntimeError(
        f"Set EE_PROJECT and GOOGLE_APPLICATION_CREDENTIALS in {_env_path} "
        "(see .env.example)."
    )
if not os.path.isabs(EE_KEY_FILE):
    EE_KEY_FILE = os.path.join(_SCRIPT_DIR, EE_KEY_FILE)


def _init_earth_engine():
    if not os.path.isfile(EE_KEY_FILE):
        raise FileNotFoundError(
            f"Service-account key not found: {EE_KEY_FILE}\n"
            f"Point GOOGLE_APPLICATION_CREDENTIALS in {_env_path} at the JSON key."
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = EE_KEY_FILE
    with open(EE_KEY_FILE) as f:
        sa_email = json.load(f)["client_email"]
    credentials = ee.ServiceAccountCredentials(sa_email, EE_KEY_FILE)
    ee.Initialize(credentials, project=GEE_PROJECT)
    print(f"Earth Engine initialized as {sa_email} on {GEE_PROJECT}.")


_init_earth_engine()

# ------------------------------------------------------------------ CONFIG
DEKAD_DAYS = 10
T_CONTEXT = 24
N_DEKADS = 24  # Set to 24 to keep only the inference window locally

ADVANCE_ARCHIVE = True

_manifest_path = f'{DOWNLOADS_DIR}/dekad_manifest.json'

if os.path.exists(_manifest_path):
    with open(_manifest_path) as f:
        _prev = json.load(f)
    _prev_end = date.fromisoformat(_prev['archive_end'])
    _gap_days = (date.today() - _prev_end).days
    _k = max(0, _gap_days // DEKAD_DAYS) if ADVANCE_ARCHIVE else 0
    ARCHIVE_END = _prev_end + timedelta(days=DEKAD_DAYS * _k)
else:
    ARCHIVE_END = date(2026, 8, 12) 

CS_THRESHOLD = 0.60
SCL_BAD = [1, 3, 8, 9, 10, 11]
NODATA_I16 = -32768
S2_SCALE = 10000.0
S1_SCALE = 100.0
SLOPE_SCALE = 100.0
DW_MISSING = 9
NO_OBS_AGE = 255

S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']

# ------------------------------------------------------------------------------
# CELL 2: AOI
# ------------------------------------------------------------------------------
geojson_data = {
  "type": "FeatureCollection",
  "features": [{
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [[
            [78.23253733075279, 17.575005952814324],
            [78.23253733075279, 17.200155099308077],
            [78.65345758223617, 17.200155099308077],
            [78.65345758223617, 17.575005952814324],
            [78.23253733075279, 17.575005952814324]
          ]],
        "type": "Polygon"
      }
    }]
}
aoi_geom = ee.FeatureCollection([ee.Feature(f) for f in geojson_data['features']]).geometry()
EXPORT_CRS = 'EPSG:32644'

# ------------------------------------------------------------------------------
# CELL 3: Build Manifest
# ------------------------------------------------------------------------------
dekads = []
for i in range(N_DEKADS):
    end = ARCHIVE_END - timedelta(days=DEKAD_DAYS * (N_DEKADS - 1 - i))
    start = end - timedelta(days=DEKAD_DAYS)
    mid = start + timedelta(days=DEKAD_DAYS // 2)
    dekads.append({
        "index": i, "label": start.isoformat(),
        "start": start.isoformat(), "end": end.isoformat(),
        "month_index": mid.month - 1,
    })

manifest = {
    "dekad_days": DEKAD_DAYS, "n_dekads": N_DEKADS, "context_length": T_CONTEXT,
    "archive_end": ARCHIVE_END.isoformat(), "crs": EXPORT_CRS,
    "scale_m": 10, "era5_scale_m": 1000, "dekads": dekads,
    "s2_bands": S2_BANDS, "s1_bands": S1_BANDS,
    "s2obs_bands": ["n_obs", "age_days"], "s1obs_bands": ["n_obs"],
    "era5_bands": ["temperature_2m", "total_precipitation_monthly_equiv"],
    "srtm_bands": ["elevation", "slope"],
    "nodata_i16": NODATA_I16, "s2_scale": S2_SCALE,
    "s1_scale": S1_SCALE, "slope_scale": SLOPE_SCALE,
    "dw_missing_class": DW_MISSING, "no_obs_age": NO_OBS_AGE,
    "cs_threshold": CS_THRESHOLD, "scl_bad_classes": SCL_BAD
}

with open(f"{DOWNLOADS_DIR}/dekad_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ------------------------------------------------------------------------------
# PRUNING LOGIC (ROLLING WINDOW)
# ------------------------------------------------------------------------------
def prune_expired_dekads(folder_path, active_dekads):
    active_labels = {d["label"] for d in active_dekads}
    prefixes = ("S2_", "S2OBS_", "S1_", "S1OBS_", "DW_", "ERA5_")
    deleted = 0
    freed = 0
    
    for fname in os.listdir(folder_path):
        if not fname.endswith(".tif") or fname.startswith("SRTM"):
            continue
        if any(fname.startswith(p) for p in prefixes):
            try:
                label = os.path.splitext(fname)[0].split("_", 1)[1]
            except IndexError:
                continue
                
            if label not in active_labels:
                fpath = os.path.join(folder_path, fname)
                try:
                    freed += os.path.getsize(fpath)
                    os.remove(fpath)
                    deleted += 1
                    print(f"  [PRUNED] Expired dekad file deleted: {fname}")
                except OSError as e:
                    print(f"  Could not delete {fname}: {e}")
                    
    if deleted > 0:
        print(f"Pruned {deleted} expired files. Freed ~{freed / 1e6:.1f} MB.")

print("\n--- CHECKING FOR EXPIRED DEKADS ---")
prune_expired_dekads(DOWNLOADS_DIR, dekads)

# ------------------------------------------------------------------------------
# CELL 4: Composite builders
# ------------------------------------------------------------------------------
CS_PLUS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

def _masked_dummy(bands):
    return ee.Image.constant([0] * len(bands)).rename(bands).selfMask().toFloat()

def build_s2(start, end):
    end_date = ee.Date(end)
    def _mask(img):
        scl = img.select('SCL')
        scl_ok = scl.remap(SCL_BAD, [0] * len(SCL_BAD), 1)
        cs_ok = img.select('cs_cdf').gte(CS_THRESHOLD)
        out = img.updateMask(scl_ok.And(cs_ok)).select(S2_BANDS)
        age = (ee.Image.constant(end_date.difference(img.date(), 'day'))
               .rename('age_days').float())
        return out.addBands(age.updateMask(out.select('B4').mask()))

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi_geom).filterDate(start, end)
           .linkCollection(CS_PLUS, ['cs_cdf']).map(_mask))

    refl = (col.select(S2_BANDS).map(lambda img: img.toFloat())
            .merge(ee.ImageCollection([_masked_dummy(S2_BANDS)]))
            .median().unmask(NODATA_I16).round().toInt16().clip(aoi_geom))
            
    n_obs = (col.select(['B4']).map(lambda img: img.toFloat())
             .merge(ee.ImageCollection([_masked_dummy(['B4'])]))
             .count().unmask(0).clamp(0, 254).toUint8().rename('n_obs'))
             
    age = (col.select(['age_days']).merge(ee.ImageCollection([_masked_dummy(['age_days'])]))
           .min().round().clamp(0, 254).unmask(NO_OBS_AGE).toUint8().rename('age_days'))
           
    return refl, ee.Image.cat([n_obs, age]).clip(aoi_geom)

def build_s1(start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi_geom).filterDate(start, end)
           .filter(ee.Filter.eq('instrumentMode', 'IW')).select(S1_BANDS))
           
    backscatter = (col.merge(ee.ImageCollection([_masked_dummy(S1_BANDS)]))
                   .median().multiply(S1_SCALE).unmask(NODATA_I16).round().toInt16().clip(aoi_geom))
                   
    n_obs = (col.select(['VV']).merge(ee.ImageCollection([_masked_dummy(['VV'])]))
             .count().unmask(0).clamp(0, 254).toUint8().rename('n_obs').clip(aoi_geom))
             
    return backscatter, n_obs

def build_era5(start, end):
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end))
    temp = col.select('temperature_2m').mean().rename('temperature_2m')
    precip = (col.select('total_precipitation_sum').sum()
              .multiply(30.44 / DEKAD_DAYS).rename('total_precipitation_monthly_equiv'))
    return ee.Image.cat([temp, precip]).toFloat().clip(aoi_geom)

def build_dw(start, end):
    col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
           .filterBounds(aoi_geom).filterDate(start, end).select(['label'])
           .map(lambda img: img.toFloat()))
    return (col.merge(ee.ImageCollection([_masked_dummy(['label'])]))
            .mode().unmask(DW_MISSING).round().toUint8().rename('label').clip(aoi_geom))

def build_srtm():
    srtm = ee.Image('USGS/SRTMGL1_003')
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation).multiply(SLOPE_SCALE).rename('slope')
    return (ee.Image.cat([elevation, slope]).unmask(NODATA_I16).round().toInt16().clip(aoi_geom))

# ------------------------------------------------------------------------------
# CELL 5: Direct Local Download
# ------------------------------------------------------------------------------
local_names = {os.path.splitext(f)[0] for f in os.listdir(DOWNLOADS_DIR) if f.endswith(".tif")}
done_names = local_names.copy()

def needs_export(name): return name not in done_names
def already_have(name): return name in done_names

def _get_geemap():
    import geemap
    return geemap

downloaded = []
failed_downloads = []

def download_local(image, name, scale=10):
    if not needs_export(name):
        return False
    geemap = _get_geemap()
    dest = os.path.join(DOWNLOADS_DIR, f"{name}.tif")
    for attempt in range(3):
        try:
            geemap.download_ee_image(
                image=image, filename=dest, region=aoi_geom,
                crs=EXPORT_CRS, scale=scale, overwrite=True,
            )
            done_names.add(name)
            downloaded.append(name)
            return True
        except Exception as e:
            if os.path.isfile(dest):
                try: os.remove(dest)
                except OSError: pass
            if attempt == 2:
                print(f"  FAILED to download {name} after 3 attempts: {e}")
                failed_downloads.append((name, str(e)))
                return False
            print(f"  download error for {name}: {e}. Retrying ({attempt + 1}/3) ...")
            time.sleep(5 * (attempt + 1))

# ---- ERA5 LATE FILL LOGIC ----
ERA5_REFRESH_LOG = os.path.join(DOWNLOADS_DIR, "era5_refresh_log.json")
def _load_era5_log():
    if os.path.exists(ERA5_REFRESH_LOG):
        with open(ERA5_REFRESH_LOG) as f: return set(json.load(f))
    return set()
def _save_era5_log(done_labels):
    with open(ERA5_REFRESH_LOG, "w") as f: json.dump(sorted(list(done_labels)), f, indent=2)

print("\n--- STARTING FULL LOCAL DOWNLOAD SYNC ---")

if needs_export("SRTM_Static"):
    download_local(build_srtm(), "SRTM_Static")
    print("  downloaded SRTM_Static" if already_have("SRTM_Static") else "  FAILED SRTM_Static")
else:
    print("  SRTM_Static already present locally")

era5_refreshed = _load_era5_log()
if len(dekads) >= 2:
    t_minus_1 = dekads[-2]["label"]
    era5_name = f"ERA5_{t_minus_1}"
    if t_minus_1 not in era5_refreshed:
        local_path = os.path.join(DOWNLOADS_DIR, f"{era5_name}.tif")
        if os.path.exists(local_path):
            os.remove(local_path)
            done_names.discard(era5_name)
            
        print(f"Executing ERA5 late-fill for previous dekad {t_minus_1}...")
        if download_local(build_era5(dekads[-2]["start"], dekads[-2]["end"]), era5_name, scale=1000):
            era5_refreshed.add(t_minus_1)
            _save_era5_log(era5_refreshed)
            print(f"  ERA5 late-fill: {era5_name} re-downloaded with complete window.")

# ---- MAIN DOWNLOAD LOOP ----
n_skipped = 0
for d in reversed(dekads):            
    lbl, start, end = d["label"], d["start"], d["end"]
    
    names_needed = [f"S2_{lbl}", f"S2OBS_{lbl}", f"S1_{lbl}", f"S1OBS_{lbl}", f"DW_{lbl}", f"ERA5_{lbl}"]
    if not any(needs_export(n) for n in names_needed):
        n_skipped += 6
        continue

    print(f"Processing dekad {lbl}...")
    s2_refl, s2_obs = build_s2(start, end)
    s1_bs, s1_obs = build_s1(start, end)

    fetched_here = [
        download_local(s2_refl, f"S2_{lbl}"),
        download_local(s2_obs, f"S2OBS_{lbl}"),
        download_local(s1_bs, f"S1_{lbl}"),
        download_local(s1_obs, f"S1OBS_{lbl}"),
        download_local(build_dw(start, end), f"DW_{lbl}"),
        download_local(build_era5(start, end), f"ERA5_{lbl}", scale=1000),
    ]
    
    n_new = sum(fetched_here)
    n_skipped += 6 - n_new
    if n_new:
        print(f"  [{lbl}] downloaded {n_new}/6  ({start} .. {end})")

print(f"\n{len(downloaded)} new file(s) downloaded. {n_skipped} file(s) already local.")