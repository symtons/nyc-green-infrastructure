"""Step 3b — Data collection from Google Earth Engine (per-borough).

Downloads for ALL 5 NYC boroughs:
  1. Landsat 9 multispectral (B, G, R, NIR) — per-borough, summer 2024
  2. Landsat 9 thermal (LST in °C) — NYC-wide (already downloaded, skipped)
  3. ESA WorldCover v200 land cover (10m) — per-borough
  4. NYC borough boundaries (from NYC Open Data)

Per-borough downloads avoid GEE's 50 MB direct-download cap.
Re-running the script is safe: existing files are skipped.
"""
import sys
from pathlib import Path

import ee
import geemap
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_green.config import load_config


# ----- Constants ----------------------------------------------------------

GEE_PROJECT_ID = "pivotal-equinox-445505-b3"

LANDSAT_DATE_START = "2024-06-01"
LANDSAT_DATE_END   = "2024-08-31"
CLOUD_THRESHOLD    = 20

# Minimum file size to consider a download valid (in bytes).
# Anything smaller than 100 KB is almost certainly a failed/empty export.
MIN_VALID_FILE_SIZE = 100_000

# NYC Open Data — borough boundaries (VERIFIED working endpoint, April 2026)
NYC_BOUNDARIES_URL_PRIMARY  = "https://data.cityofnewyork.us/resource/gthc-hcne.geojson"
NYC_BOUNDARIES_URL_FALLBACK = (
    "https://raw.githubusercontent.com/nycehs/NYC_geography/master/borough.geo.json"
)

# Which boroughs to pull (matches boro_name field in NYC Open Data)
BOROUGHS = ["Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]


# ----- Helpers ------------------------------------------------------------

def slug(borough_name: str) -> str:
    """Manhattan -> manhattan, Staten Island -> staten_island."""
    return borough_name.lower().replace(" ", "_")


def verify_download(path: Path, label: str):
    """Raise if a file doesn't exist or is suspiciously small."""
    if not path.exists():
        raise RuntimeError(f"{label}: file not created at {path}")
    size = path.stat().st_size
    if size < MIN_VALID_FILE_SIZE:
        path.unlink()  # delete the garbage file so re-runs retry
        raise RuntimeError(
            f"{label}: downloaded file is suspiciously small ({size} bytes). "
            f"Deleted. Likely the GEE export silently failed."
        )
    print(f"  ✓ {label}: {size / 1_000_000:.1f} MB")


def mask_landsat_clouds(image):
    """Mask clouds/shadows/snow/dilated clouds from QA_PIXEL bitmask."""
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    dilated = qa.bitwiseAnd(1 << 1).eq(0)
    return image.updateMask(cloud.And(shadow).And(snow).And(dilated))


def scale_landsat_sr(image):
    """Apply Collection 2 SR and thermal scaling factors."""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)
    return image.addBands(optical, None, True).addBands(thermal, None, True)


# ----- Downloads ----------------------------------------------------------

def download_nyc_boundaries(out_path: Path):
    """Fetch all 5 borough polygons from NYC Open Data."""
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"  ↳ already exists, skipping: {out_path.name}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for url in (NYC_BOUNDARIES_URL_PRIMARY, NYC_BOUNDARIES_URL_FALLBACK):
        try:
            print(f"  Trying: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_path.write_text(resp.text, encoding="utf-8")
            print(f"  ✓ Saved boundaries from {url}")
            return
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    raise RuntimeError("All boundary sources failed")


def get_borough_geometry(boundaries_path: Path, borough: str) -> ee.Geometry:
    """Load borough polygon from the NYC boundaries GeoJSON.

    Tolerant of different property naming conventions across NYC Open Data
    variants: boro_name, boroname, BoroName, name, etc. Also supports
    numeric boro_code (1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island).
    """
    import json

    gj = json.loads(boundaries_path.read_text(encoding="utf-8"))

    code_map = {
        "manhattan": "1",
        "bronx": "2",
        "brooklyn": "3",
        "queens": "4",
        "staten island": "5",
    }
    target_name = borough.lower()
    target_code = code_map.get(target_name)

    name_fields = ["boro_name", "boroname", "BoroName", "BoroughName", "name", "NAME"]
    code_fields = ["boro_code", "borocode", "BoroCode"]

    for feat in gj["features"]:
        props = feat.get("properties", {})
        for f in name_fields:
            if f in props and str(props[f]).lower() == target_name:
                return ee.Geometry(feat["geometry"])
        if target_code is not None:
            for f in code_fields:
                if f in props and str(props[f]) == target_code:
                    return ee.Geometry(feat["geometry"])

    sample_props = gj["features"][0].get("properties", {}) if gj["features"] else {}
    raise KeyError(
        f"Borough '{borough}' not found in boundaries GeoJSON. "
        f"Available property keys on first feature: {list(sample_props.keys())}"
    )

def download_landsat_multispectral(roi, out_path: Path, borough_label: str):
    if out_path.exists() and out_path.stat().st_size > MIN_VALID_FILE_SIZE:
        print(f"  ↳ already exists, skipping: {out_path.name}")
        return

    print(f"  Building Landsat 9 multispectral composite for {borough_label}...")
    collection = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(LANDSAT_DATE_START, LANDSAT_DATE_END)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_THRESHOLD))
        .map(mask_landsat_clouds)
        .map(scale_landsat_sr)
    )
    n = collection.size().getInfo()
    print(f"  Scenes: {n}")
    if n == 0:
        raise RuntimeError(f"No Landsat scenes matched filters for {borough_label}")

    composite = (
        collection.median()
        .select(["SR_B2", "SR_B3", "SR_B4", "SR_B5"],
                ["blue", "green", "red", "nir"])
        .clip(roi)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {out_path.name}...")
    geemap.ee_export_image(
        composite,
        filename=str(out_path),
        scale=30,
        region=roi,
        crs="EPSG:4326",
        file_per_band=False,
    )
    verify_download(out_path, f"Landsat {borough_label}")


def download_worldcover_borough(roi, out_path: Path, borough_label: str):
    if out_path.exists() and out_path.stat().st_size > MIN_VALID_FILE_SIZE:
        print(f"  ↳ already exists, skipping: {out_path.name}")
        return

    print(f"  Fetching WorldCover for {borough_label}...")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().clip(roi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {out_path.name}...")
    geemap.ee_export_image(
        wc,
        filename=str(out_path),
        scale=10,
        region=roi,
        crs="EPSG:4326",
        file_per_band=False,
    )
    verify_download(out_path, f"WorldCover {borough_label}")


# ----- Main ---------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3b — DATA COLLECTION (PER-BOROUGH)")
    print("=" * 60)

    ee.Initialize(project=GEE_PROJECT_ID)
    print(f"✓ Earth Engine initialized ({GEE_PROJECT_ID})")

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    print(f"✓ Raw data directory: {raw_dir}")
    print()

    # --- Step 1: NYC borough boundaries (needed to build per-borough geometries)
    print("[1/4] NYC borough boundaries")
    boundaries_path = raw_dir / "boundaries" / "nyc_boroughs.geojson"
    download_nyc_boundaries(boundaries_path)
    print()

    # --- Step 2: LST (already downloaded as NYC-wide file, just verify)
    print("[2/4] Landsat 9 LST — NYC-wide (from previous run)")
    lst_path = raw_dir / "landsat" / "nyc_lst.tif"
    if lst_path.exists() and lst_path.stat().st_size > MIN_VALID_FILE_SIZE:
        print(f"  ✓ Already present: {lst_path.name} "
              f"({lst_path.stat().st_size / 1_000_000:.1f} MB)")
    else:
        print(f"  ✗ Missing. Re-running LST download...")
        # If missing, fall back to the merged NYC download — it worked before
        nyc_bbox = ee.Geometry.Rectangle([-74.2591, 40.4774, -73.7004, 40.9176])
        collection = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(nyc_bbox)
            .filterDate(LANDSAT_DATE_START, LANDSAT_DATE_END)
            .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_THRESHOLD))
            .map(mask_landsat_clouds)
            .map(scale_landsat_sr)
        )
        lst = (collection.median().select("ST_B10").subtract(273.15)
               .rename("lst_celsius").clip(nyc_bbox))
        lst_path.parent.mkdir(parents=True, exist_ok=True)
        geemap.ee_export_image(
            lst, filename=str(lst_path), scale=30, region=nyc_bbox,
            crs="EPSG:4326", file_per_band=False,
        )
        verify_download(lst_path, "LST NYC-wide")
    print()

    # --- Step 3: Per-borough Landsat multispectral
    print("[3/4] Per-borough Landsat 9 multispectral")
    for borough in BOROUGHS:
        print(f"\n  --- {borough} ---")
        geom = get_borough_geometry(boundaries_path, borough)
        out_path = raw_dir / "landsat" / f"{slug(borough)}_landsat.tif"
        download_landsat_multispectral(geom, out_path, borough)
    print()

    # --- Step 4: Per-borough WorldCover
    print("[4/4] Per-borough ESA WorldCover")
    for borough in BOROUGHS:
        print(f"\n  --- {borough} ---")
        geom = get_borough_geometry(boundaries_path, borough)
        out_path = raw_dir / "landcover" / f"{slug(borough)}_worldcover.tif"
        download_worldcover_borough(geom, out_path, borough)
    print()

    print("=" * 60)
    print("✓ DATA COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()