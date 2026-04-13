"""Step 8 — Equity layer: NYC Heat Vulnerability Index rasterized to 30m grid.

The HVI dataset (NYC Open Data id 4mhf-duep) is indexed by ZCTA (ZIP Code
Tabulation Area) and only contains {zcta20, hvi} — no geometry. We join it
to NYC Modified ZCTA polygons (MODZCTA, id pri4-ifjk) which do have geometry.

Output: data/interim/nyc_equity_30m.tif
  - Single-band float32 raster on the same 30m UTM grid as the other layers
  - Values are HVI score rescaled from 1–5 → 0–100 so they compose cleanly
    with the other 0–100 priority components
  - NaN outside the NYC boundary (inherited from the reference raster)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize


HVI_JSON_URL = "https://data.cityofnewyork.us/resource/4mhf-duep.json"

# MODZCTA geospatial export (this URL preserves attribute fields; the /resource/
# endpoint strips them and returns geometry-only)
MODZCTA_URL = (
    "https://data.cityofnewyork.us/api/geospatial/pri4-ifjk"
    "?method=export&format=GeoJSON"
)


# ==========================================================================
# Download & join
# ==========================================================================

def download_hvi_joined(out_path: Path) -> gpd.GeoDataFrame:
    """Fetch HVI table + MODZCTA polygons, join them, cache as GeoJSON."""
    if out_path.exists() and out_path.stat().st_size > 5000:
        print(f"  ↳ already exists, skipping download: {out_path.name}")
        return gpd.read_file(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- HVI table: zcta20, hvi ---
    print(f"  Fetching HVI table: {HVI_JSON_URL}")
    resp = requests.get(HVI_JSON_URL, timeout=60)
    resp.raise_for_status()
    hvi_df = pd.DataFrame(resp.json())
    print(f"  HVI rows: {len(hvi_df)}  columns: {list(hvi_df.columns)}")
    if "zcta20" not in hvi_df.columns or "hvi" not in hvi_df.columns:
        raise RuntimeError(
            f"Unexpected HVI schema: {list(hvi_df.columns)}. Expected zcta20 and hvi."
        )
    # Normalize: make zcta20 a clean string, hvi numeric
    hvi_df["zcta20"] = hvi_df["zcta20"].astype(str).str.strip()
    hvi_df["hvi"] = pd.to_numeric(hvi_df["hvi"], errors="coerce")

    # --- MODZCTA polygons ---
    print(f"  Fetching MODZCTA polygons: {MODZCTA_URL}")
    modzcta = gpd.read_file(MODZCTA_URL)
    print(f"  MODZCTA features: {len(modzcta)}  columns: {list(modzcta.columns)}")

    # Find the MODZCTA code column (should be "modzcta", but be tolerant)
    code_candidates = ["modzcta", "MODZCTA", "zcta", "zip", "zipcode"]
    modzcta_col = next((c for c in code_candidates if c in modzcta.columns), None)
    if modzcta_col is None:
        raise RuntimeError(
            f"Could not find ZCTA code column in MODZCTA data. "
            f"Available: {list(modzcta.columns)}"
        )
    print(f"  Joining on HVI.zcta20 = MODZCTA.{modzcta_col}")

    modzcta[modzcta_col] = modzcta[modzcta_col].astype(str).str.strip()

    joined = modzcta.merge(
        hvi_df,
        left_on=modzcta_col,
        right_on="zcta20",
        how="left",
    )
    n_matched = joined["hvi"].notna().sum()
    print(f"  Matched features: {n_matched}/{len(joined)}")
    if n_matched == 0:
        # Diagnose the mismatch
        sample_hvi = hvi_df["zcta20"].head(3).tolist()
        sample_mod = modzcta[modzcta_col].head(3).tolist()
        raise RuntimeError(
            f"No HVI rows matched any MODZCTA polygon.\n"
            f"  HVI zcta20 samples: {sample_hvi}\n"
            f"  MODZCTA {modzcta_col} samples: {sample_mod}"
        )

    joined.to_file(out_path, driver="GeoJSON")
    return joined


# ==========================================================================
# Rasterization
# ==========================================================================

def rasterize_hvi(
    gdf: gpd.GeoDataFrame,
    reference_raster_path: Path,
    out_path: Path,
) -> dict:
    """Rasterize HVI polygons to the reference grid, rescaling 1–5 → 0–100."""
    with rasterio.open(reference_raster_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    # Reproject polygons into the reference CRS (UTM 18N)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(ref_crs)

    # Filter to rows that have a valid HVI score and geometry
    valid = (
        gdf["hvi"].notna()
        & gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
    )
    gdf_valid = gdf[valid].copy()
    print(f"  Rasterizing {len(gdf_valid)} polygons (of {len(gdf)} total)...")

    # Scale HVI 1..5 -> 0..100
    scaled = ((gdf_valid["hvi"].astype(float) - 1.0) / 4.0 * 100.0).clip(0, 100)
    shapes = list(zip(gdf_valid.geometry.values, scaled.astype(np.float32).values))

    raster = rasterize(
        shapes=shapes,
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=np.nan,
        dtype="float32",
        all_touched=False,
    )

    # Mask to NYC boundary using the reference raster's nodata
    with rasterio.open(reference_raster_path) as ref:
        ref_band = ref.read(1)
    if np.issubdtype(ref_band.dtype, np.floating):
        mask_outside = np.isnan(ref_band)
    else:
        nd = ref.nodata if ref.nodata is not None else 0
        mask_outside = ref_band == nd
    raster[mask_outside] = np.nan

    out_profile = ref_profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(raster, 1)

    valid_vals = raster[~np.isnan(raster)]
    return {
        "out_path": str(out_path),
        "shape": list(raster.shape),
        "n_polygons_rasterized": int(len(shapes)),
        "valid_pixels": int(valid_vals.size),
        "score_min": float(valid_vals.min()) if valid_vals.size > 0 else None,
        "score_max": float(valid_vals.max()) if valid_vals.size > 0 else None,
        "score_mean": float(valid_vals.mean()) if valid_vals.size > 0 else None,
    }