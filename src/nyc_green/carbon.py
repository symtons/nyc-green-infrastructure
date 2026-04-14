"""Carbon sequestration accounting for NYC's current vegetation.

Computes the total annual CO2 sequestration from existing NYC vegetation,
tiered by NDVI density. This is a status-quo calculation, not a scenario —
it reports what the city's current green cover absorbs per year, with no
intervention or "potential" framing.

NDVI density thresholds and per-hectare rates are from the i-Tree Canopy
literature, which validates these numbers against field measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio


# --- Rates (from i-Tree Canopy literature) -------------------------------

NDVI_DENSE_MIN     = 0.6
NDVI_MODERATE_MIN  = 0.3
NDVI_SPARSE_MIN    = 0.1

RATE_DENSE_T_PER_HA    = 10.0   # tCO2/ha/yr, NDVI >= 0.6
RATE_MODERATE_T_PER_HA = 5.0    # tCO2/ha/yr, 0.3 <= NDVI < 0.6
RATE_SPARSE_T_PER_HA   = 2.0    # tCO2/ha/yr, 0.1 <= NDVI < 0.3

PIXEL_AREA_M2 = 900.0  # 30m grid
PIXEL_AREA_HA = PIXEL_AREA_M2 / 10_000.0


# --- Real-world equivalent conversion factors ----------------------------
# Per-unit annual carbon equivalents for scale communication

T_PER_CAR_PER_YEAR       = 4.6     # average passenger vehicle
T_PER_TREE_SEEDLING_10YR = 0.06    # tree seedling grown for 10 years
T_PER_HOME_ENERGY        = 7.5     # single-family home annual energy use


def load_landsat_and_landcover(
    landsat_path: Path,
    landcover_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the aligned Landsat (for NDVI) and the land cover prediction.

    Returns (ndvi, veg_mask, valid_mask):
      ndvi      — float32 array of NDVI values
      veg_mask  — bool array, True where the land cover says "vegetation"
      valid_mask — bool array, True where both NDVI and land cover are valid
    """
    with rasterio.open(landsat_path) as src:
        red = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)

    # NDVI
    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)

    # Land cover: class 0 = vegetation in our 3-class taxonomy
    with rasterio.open(landcover_path) as src:
        lc = src.read(1)
        nodata = src.nodata if src.nodata is not None else 255
    veg_mask = (lc == 0)

    # Valid where both NDVI is finite and the land cover is not nodata
    valid = np.isfinite(ndvi) & (lc != nodata) & (~np.isnan(red)) & (~np.isnan(nir))
    return ndvi, veg_mask, valid


def compute_carbon_summary(
    landsat_path: Path,
    landcover_path: Path,
) -> dict:
    """Compute the full carbon sequestration accounting for NYC.

    Returns a dict with per-tier areas, per-tier carbon, totals, and
    real-world equivalents. Everything in this dict is fact-checkable
    against the underlying rasters.
    """
    ndvi, veg_mask, valid = load_landsat_and_landcover(
        landsat_path=landsat_path,
        landcover_path=landcover_path,
    )

    # Only count vegetation pixels with valid NDVI
    valid_veg = veg_mask & valid
    ndvi_veg = ndvi[valid_veg]

    # Tier the vegetation pixels by NDVI
    dense_mask    = ndvi_veg >= NDVI_DENSE_MIN
    moderate_mask = (ndvi_veg >= NDVI_MODERATE_MIN) & (ndvi_veg < NDVI_DENSE_MIN)
    sparse_mask   = (ndvi_veg >= NDVI_SPARSE_MIN)   & (ndvi_veg < NDVI_MODERATE_MIN)
    below_sparse  = ndvi_veg < NDVI_SPARSE_MIN

    n_dense    = int(dense_mask.sum())
    n_moderate = int(moderate_mask.sum())
    n_sparse   = int(sparse_mask.sum())
    n_below    = int(below_sparse.sum())

    # Areas (hectares)
    area_dense_ha    = n_dense    * PIXEL_AREA_HA
    area_moderate_ha = n_moderate * PIXEL_AREA_HA
    area_sparse_ha   = n_sparse   * PIXEL_AREA_HA
    area_below_ha    = n_below    * PIXEL_AREA_HA
    area_total_ha    = area_dense_ha + area_moderate_ha + area_sparse_ha + area_below_ha

    # Carbon (tCO2/year) — below-sparse gets rate 0
    carbon_dense    = area_dense_ha    * RATE_DENSE_T_PER_HA
    carbon_moderate = area_moderate_ha * RATE_MODERATE_T_PER_HA
    carbon_sparse   = area_sparse_ha   * RATE_SPARSE_T_PER_HA
    carbon_total    = carbon_dense + carbon_moderate + carbon_sparse

    # Real-world equivalents
    equiv_cars         = carbon_total / T_PER_CAR_PER_YEAR
    equiv_tree_seedlings_10yr = carbon_total / T_PER_TREE_SEEDLING_10YR
    equiv_homes        = carbon_total / T_PER_HOME_ENERGY

    tiers = [
        {
            "name": "Dense vegetation",
            "ndvi_range": f"NDVI >= {NDVI_DENSE_MIN}",
            "pixels": n_dense,
            "area_ha": round(area_dense_ha, 2),
            "rate_t_per_ha_yr": RATE_DENSE_T_PER_HA,
            "carbon_t_per_yr": round(carbon_dense, 1),
        },
        {
            "name": "Moderate vegetation",
            "ndvi_range": f"{NDVI_MODERATE_MIN} <= NDVI < {NDVI_DENSE_MIN}",
            "pixels": n_moderate,
            "area_ha": round(area_moderate_ha, 2),
            "rate_t_per_ha_yr": RATE_MODERATE_T_PER_HA,
            "carbon_t_per_yr": round(carbon_moderate, 1),
        },
        {
            "name": "Sparse vegetation",
            "ndvi_range": f"{NDVI_SPARSE_MIN} <= NDVI < {NDVI_MODERATE_MIN}",
            "pixels": n_sparse,
            "area_ha": round(area_sparse_ha, 2),
            "rate_t_per_ha_yr": RATE_SPARSE_T_PER_HA,
            "carbon_t_per_yr": round(carbon_sparse, 1),
        },
    ]

    return {
        "tiers": tiers,
        "totals": {
            "vegetation_area_ha": round(area_total_ha, 2),
            "carbon_t_per_yr": round(carbon_total, 1),
            "area_excluded_low_ndvi_ha": round(area_below_ha, 2),
        },
        "equivalents": {
            "cars_per_year":        round(equiv_cars),
            "tree_seedlings_10yr":  round(equiv_tree_seedlings_10yr),
            "homes_per_year":       round(equiv_homes),
        },
        "methodology": {
            "ndvi_source": "Landsat 9 red/NIR bands, summer 2024 median composite",
            "vegetation_source": "U-Net predicted land cover, class 0 = vegetation",
            "pixel_area_m2": PIXEL_AREA_M2,
            "rates_source": "i-Tree Canopy literature (validated urban forestry rates)",
            "equivalents_source": "EPA greenhouse gas equivalencies (cars, trees, homes)",
        },
    }