"""Step 9 — Priority scoring for green infrastructure investment.

Combines four components into a single 0–100 priority score per pixel:
  - heat (25%)              from Landsat thermal LST
  - vegetation deficit (30%) from NDVI (inverted)
  - built-up (20%)          from land cover classification
  - equity (25%)            from NYC Heat Vulnerability Index

All four components are rescaled to 0–100 before weighting. Water pixels
are excluded from scoring entirely (can't plant trees in the river).

Outputs:
  - priority_score.tif    float32, 0–100, NaN where excluded
  - priority_zones.tif    uint8, categorical: 0=none, 1=low, 2=mod, 3=high, 4=critical
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


# Priority category codes
PRIORITY_NONE     = 0
PRIORITY_LOW      = 1
PRIORITY_MODERATE = 2
PRIORITY_HIGH     = 3
PRIORITY_CRITICAL = 4


# ==========================================================================
# Component computation
# ==========================================================================

def compute_heat_component(lst_celsius: np.ndarray) -> np.ndarray:
    """Rescale land surface temperature to a 0–100 heat priority score.

    Uses a piecewise-linear mapping calibrated to NYC summer LST:
      LST <= 20°C  -> 0   (cool, no priority)
      LST == 30°C  -> 50  (moderate)
      LST >= 40°C  -> 100 (extreme, maximum priority)
    Linear interpolation between those anchor points.
    """
    lst = lst_celsius.astype(np.float32)
    score = np.full_like(lst, np.nan, dtype=np.float32)

    valid = ~np.isnan(lst)
    # Linear rescale from [20, 40] -> [0, 100]
    score[valid] = np.clip((lst[valid] - 20.0) / (40.0 - 20.0) * 100.0, 0, 100)
    return score


def compute_vegetation_deficit_component(
    red_band: np.ndarray,
    nir_band: np.ndarray,
) -> np.ndarray:
    """NDVI-derived vegetation deficit score, 0–100.

    NDVI ranges roughly [-0.1, 0.9] over NYC. We invert it so that:
      NDVI <= 0.0  -> 100 (no vegetation, max priority)
      NDVI == 0.4  -> 50  (moderate)
      NDVI >= 0.7  -> 0   (dense vegetation, no priority)
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denom = nir + red + 1e-8
    ndvi = np.clip((nir - red) / denom, -1.0, 1.0)

    score = np.full_like(ndvi, np.nan, dtype=np.float32)
    valid = ~np.isnan(ndvi)
    # Piecewise: NDVI 0.0 -> 100, NDVI 0.7 -> 0
    score[valid] = np.clip((0.7 - ndvi[valid]) / 0.7 * 100.0, 0, 100)
    return score, ndvi

def compute_builtup_component(
    landcover: np.ndarray,
    source: str,
    built_up_class: int = 2,
    vegetation_class: int = 0,
    water_class: int = 1,
    nodata: int = 255,
) -> np.ndarray:
    """Built-up score: high where built-up, low where vegetation, excluded on water.

    `source` specifies which taxonomy the landcover array is in:
      - 'model'      → already our 3-class taxonomy (0=veg, 1=water, 2=built)
      - 'worldcover' → raw ESA WorldCover codes (10/20/30, 40, 50, 60/70, 80/90/95, 100);
                       reclassified on the fly to our 3-class taxonomy.

    Returns 0–100 float array with NaN on water, nodata, and unknown pixels.
    """
    if source == "worldcover":
        from nyc_green.tiles import reclassify_worldcover
        lc3 = reclassify_worldcover(landcover)
    elif source == "model":
        lc3 = landcover
    else:
        raise ValueError(f"Unknown landcover source: {source!r}")

    score = np.full(lc3.shape, np.nan, dtype=np.float32)
    score[lc3 == built_up_class] = 100.0
    score[lc3 == vegetation_class] = 20.0
    # Water and nodata (255) stay NaN — we don't plant trees on water
    return score

# ==========================================================================
# Combination
# ==========================================================================

def combine_priority_score(
    heat: np.ndarray,
    veg_deficit: np.ndarray,
    built_up: np.ndarray,
    equity: np.ndarray,
    weights: dict,
) -> np.ndarray:
    """Weighted combination of four components into a single priority score.

    Each input is a 0–100 float array, possibly with NaN values.
    Output is 0–100 float, NaN where ANY required component is NaN.

    weights: dict with keys 'heat', 'vegetation_deficit', 'built_up', 'equity'
             whose values sum to 1.0.
    """
    components = np.stack([heat, veg_deficit, built_up, equity], axis=0)
    w = np.array([
        weights["heat"],
        weights["vegetation_deficit"],
        weights["built_up"],
        weights["equity"],
    ], dtype=np.float32)

    # Any NaN in any component => overall NaN (strict, documented behavior)
    valid = ~np.any(np.isnan(components), axis=0)
    score = np.full(components.shape[1:], np.nan, dtype=np.float32)

    # Weighted sum on valid pixels
    weighted = np.tensordot(w, components, axes=(0, 0))
    score[valid] = weighted[valid]
    return score


def categorize_priority(
    score: np.ndarray,
    thresholds: dict,
) -> tuple[np.ndarray, dict]:
    """Bin continuous 0–100 priority into 5 categories.

    If `thresholds` contains a "mode" key equal to "percentile", it must also
    contain "percentiles" — a dict with critical/high/moderate/low as
    percentile floats (e.g. {critical: 95, high: 85, moderate: 65, low: 40}).
    The actual score cutoffs are computed from the valid pixels of `score`.

    Otherwise `thresholds` is interpreted as absolute score cutoffs
    (backwards compatible with the original 80/60/40/20 config).

    Returns (categories_array, cutoffs_used).
    """
    cat = np.full(score.shape, PRIORITY_NONE, dtype=np.uint8)
    valid_mask = ~np.isnan(score)
    valid_scores = score[valid_mask]

    if valid_scores.size == 0:
        return cat, {"critical": 100, "high": 100, "moderate": 100, "low": 100}

    mode = thresholds.get("mode", "absolute")

    if mode == "percentile":
        pcts = thresholds["percentiles"]
        cutoffs = {
            "critical": float(np.percentile(valid_scores, pcts["critical"])),
            "high":     float(np.percentile(valid_scores, pcts["high"])),
            "moderate": float(np.percentile(valid_scores, pcts["moderate"])),
            "low":      float(np.percentile(valid_scores, pcts["low"])),
        }
    else:
        cutoffs = {
            "critical": float(thresholds["critical"]),
            "high":     float(thresholds["high"]),
            "moderate": float(thresholds["moderate"]),
            "low":      float(thresholds["low"]),
        }

    t_low = cutoffs["low"]
    t_mod = cutoffs["moderate"]
    t_high = cutoffs["high"]
    t_crit = cutoffs["critical"]

    valid = valid_mask  # shorthand
    cat[valid & (score >= t_low)  & (score < t_mod)]  = PRIORITY_LOW
    cat[valid & (score >= t_mod)  & (score < t_high)] = PRIORITY_MODERATE
    cat[valid & (score >= t_high) & (score < t_crit)] = PRIORITY_HIGH
    cat[valid & (score >= t_crit)]                    = PRIORITY_CRITICAL

    return cat, cutoffs
# ==========================================================================
# Stats
# ==========================================================================

def summarize_priority_zones(categories: np.ndarray, pixel_area_m2: float = 900.0) -> dict:
    """Count pixels in each priority category and convert to hectares."""
    names = {
        PRIORITY_NONE:     "None / Excluded",
        PRIORITY_LOW:      "Low",
        PRIORITY_MODERATE: "Moderate",
        PRIORITY_HIGH:     "High",
        PRIORITY_CRITICAL: "Critical",
    }
    out = {}
    total = categories.size
    for code, name in names.items():
        count = int((categories == code).sum())
        area_ha = count * pixel_area_m2 / 10_000.0
        out[name] = {
            "code": int(code),
            "pixels": count,
            "percent": 100.0 * count / total if total > 0 else 0.0,
            "area_ha": round(area_ha, 2),
        }
    return out


def summarize_score_stats(score: np.ndarray) -> dict:
    """Descriptive stats for the continuous priority score (valid pixels only)."""
    valid = score[~np.isnan(score)]
    if valid.size == 0:
        return {"valid_pixels": 0}
    return {
        "valid_pixels": int(valid.size),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "p25": float(np.percentile(valid, 25)),
        "p75": float(np.percentile(valid, 75)),
    }


# ==========================================================================
# I/O
# ==========================================================================

def load_aligned_rasters(paths: dict) -> dict:
    """Load all aligned rasters and return as a dict of numpy arrays."""
    rasters = {}
    with rasterio.open(paths["landsat"]) as src:
        landsat = src.read()  # (4, H, W): blue, green, red, nir
        rasters["profile"] = src.profile.copy()
        rasters["blue"] = landsat[0]
        rasters["green"] = landsat[1]
        rasters["red"] = landsat[2]
        rasters["nir"] = landsat[3]
    with rasterio.open(paths["lst"]) as src:
        rasters["lst"] = src.read(1)
    with rasterio.open(paths["landcover"]) as src:
        lc = src.read(1)
        # Treat nodata as an explicit sentinel so downstream comparisons work
        nd = src.nodata
        if nd is not None:
            lc = lc.copy()
            lc[lc == nd] = 255
        rasters["landcover"] = lc
    with rasterio.open(paths["equity"]) as src:
        rasters["equity"] = src.read(1)
    return rasters


def write_score_raster(score: np.ndarray, ref_profile: dict, out_path: Path):
    profile = ref_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(score.astype(np.float32), 1)


def write_zones_raster(zones: np.ndarray, ref_profile: dict, out_path: Path):
    profile = ref_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(zones.astype(np.uint8), 1)