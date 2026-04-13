"""Step 9 entry point — compute priority scores from both land cover sources.

Produces two priority maps (one per land cover source) and compares them,
so we can report how robust the conclusions are to the choice of land cover.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.priority import (
    load_aligned_rasters,
    compute_heat_component,
    compute_vegetation_deficit_component,
    compute_builtup_component,
    combine_priority_score,
    categorize_priority,
    summarize_priority_zones,
    summarize_score_stats,
    write_score_raster,
    write_zones_raster,
)


def run_for_landcover(
    rasters: dict,
    lc_label: str,
    weights: dict,
    thresholds: dict,
    out_dir: Path,
) -> dict:
    """Compute priority maps for a single land cover source and write outputs."""
    print(f"\n[{lc_label}] Computing components...")

    heat = compute_heat_component(rasters["lst"])
    print(f"  Heat: valid={(~np.isnan(heat)).sum():,}  mean={np.nanmean(heat):.1f}")

    veg_deficit, ndvi = compute_vegetation_deficit_component(rasters["red"], rasters["nir"])
    print(f"  Veg deficit: valid={(~np.isnan(veg_deficit)).sum():,}  mean={np.nanmean(veg_deficit):.1f}")
    print(f"  (NDVI: min={np.nanmin(ndvi):.2f}  max={np.nanmax(ndvi):.2f}  mean={np.nanmean(ndvi):.2f})")

    built = compute_builtup_component(rasters["landcover"], source=lc_label)
    print(f"  Built-up: valid={(~np.isnan(built)).sum():,}  mean={np.nanmean(built):.1f}")

    equity = rasters["equity"]
    print(f"  Equity: valid={(~np.isnan(equity)).sum():,}  mean={np.nanmean(equity):.1f}")

    print(f"\n[{lc_label}] Combining with weights: {weights}")
    score = combine_priority_score(heat, veg_deficit, built, equity, weights)

    stats = summarize_score_stats(score)
    print(f"  Priority score: valid={stats['valid_pixels']:,}  "
          f"mean={stats.get('mean', 0):.1f}  "
          f"median={stats.get('median', 0):.1f}  "
          f"p25={stats.get('p25', 0):.1f}  "
          f"p75={stats.get('p75', 0):.1f}")

    print(f"\n[{lc_label}] Categorizing into priority zones...")
    zones, cutoffs = categorize_priority(score, thresholds)
    print(f"  Cutoffs used: "
          f"critical≥{cutoffs['critical']:.1f}  "
          f"high≥{cutoffs['high']:.1f}  "
          f"moderate≥{cutoffs['moderate']:.1f}  "
          f"low≥{cutoffs['low']:.1f}")
    zone_summary = summarize_priority_zones(zones, pixel_area_m2=900.0)
    for name, d in zone_summary.items():
        print(f"  {name:16s} {d['pixels']:>9,} px  ({d['percent']:5.2f}%)  {d['area_ha']:>8,.1f} ha")

    # Write outputs
    score_path = out_dir / f"priority_score_{lc_label}.tif"
    zones_path = out_dir / f"priority_zones_{lc_label}.tif"
    write_score_raster(score, rasters["profile"], score_path)
    write_zones_raster(zones, rasters["profile"], zones_path)
    print(f"  ✓ {score_path.name}")
    print(f"  ✓ {zones_path.name}")

    return {
        "label": lc_label,
        "weights": weights,
        "cutoffs_used": cutoffs,
        "score_stats": stats,
        "zone_summary": zone_summary,
        "score_path": str(score_path),
        "zones_path": str(zones_path),
    }


def compare_priority_maps(result_a: dict, result_b: dict, out_dir: Path) -> dict:
    """Compare two priority zone maps from different land cover sources.

    Both rasters may contain pixels that are nodata (255) or uncategorized (0).
    We compare pixels where BOTH maps have a valid categorical assignment
    (0 through 4 — including PRIORITY_NONE, since 'neither map considers this
    pixel priority' is itself useful agreement information).
    """
    import rasterio

    with rasterio.open(result_a["zones_path"]) as src:
        zones_a = src.read(1)
        nodata_a = src.nodata
    with rasterio.open(result_b["zones_path"]) as src:
        zones_b = src.read(1)
        nodata_b = src.nodata

    # Valid = both rasters have a non-nodata value (i.e., inside NYC for both)
    valid = np.ones_like(zones_a, dtype=bool)
    if nodata_a is not None:
        valid &= (zones_a != nodata_a)
    if nodata_b is not None:
        valid &= (zones_b != nodata_b)

    total = int(valid.sum())

    # Default result shape — always has all keys, zero values if no overlap
    result = {
        "total_compared": total,
        "exact_match": 0,
        "exact_agreement_pct": 0.0,
        "within_one_category": 0,
        "within_one_pct": 0.0,
        "pair_counts": {},
    }

    if total == 0:
        return result

    match = (zones_a == zones_b) & valid
    exact_match = int(match.sum())
    result["exact_match"] = exact_match
    result["exact_agreement_pct"] = 100.0 * exact_match / total

    diff = np.abs(zones_a.astype(np.int16) - zones_b.astype(np.int16))
    within_one = int(((diff <= 1) & valid).sum())
    result["within_one_category"] = within_one
    result["within_one_pct"] = 100.0 * within_one / total

    # Co-occurrence matrix: count pairs (a, b) where both are valid
    pair_counts = {}
    for a in range(5):
        for b in range(5):
            c = int(((zones_a == a) & (zones_b == b) & valid).sum())
            if c > 0:
                pair_counts[f"a={a},b={b}"] = c
    result["pair_counts"] = pair_counts

    return result

def main():
    print("=" * 60)
    print("STEP 9 — PRIORITY SCORING (dual land cover comparison)")
    print("=" * 60)

    cfg = load_config()
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    maps_dir = outputs_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    weights = cfg["priority_weights"]
    thresholds = cfg["priority_thresholds"]
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-6, f"weights must sum to 1.0, got {total_w}"
    print(f"Weights: {weights}  (sum={total_w})")
    print(f"Thresholds: {thresholds}")

    # --- Run 1: WorldCover as land cover source ---
    print("\n" + "=" * 60)
    print("RUN 1: ESA WorldCover as land cover source")
    print("=" * 60)
    paths_wc = {
        "landsat":   interim_dir / "nyc_landsat_30m.tif",
        "lst":       interim_dir / "nyc_lst_30m.tif",
        "landcover": interim_dir / "nyc_worldcover_30m.tif",
        "equity":    interim_dir / "nyc_equity_30m.tif",
    }
    rasters_wc = load_aligned_rasters(paths_wc)
    result_wc = run_for_landcover(rasters_wc, "worldcover", weights, thresholds, maps_dir)

    # --- Run 2: Trained U-Net predictions as land cover source ---
    print("\n" + "=" * 60)
    print("RUN 2: Trained U-Net predictions as land cover source")
    print("=" * 60)
    paths_model = dict(paths_wc)
    paths_model["landcover"] = processed_dir / "nyc_landcover_predicted.tif"
    rasters_model = load_aligned_rasters(paths_model)
    result_model = run_for_landcover(rasters_model, "model", weights, thresholds, maps_dir)

    # --- Compare the two priority maps ---
    print("\n" + "=" * 60)
    print("COMPARISON: WorldCover vs Model priority zones")
    print("=" * 60)
    comparison = compare_priority_maps(result_wc, result_model, maps_dir)
    print(f"  Total pixels compared: {comparison['total_compared']:,}")
    print(f"  Exact category agreement: {comparison['exact_agreement_pct']:.2f}%")
    print(f"  Within 1 category agreement: {comparison['within_one_pct']:.2f}%")

    # Write a combined JSON summary
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "weights": weights,
        "thresholds": thresholds,
        "runs": {
            "worldcover": result_wc,
            "model": result_model,
        },
        "comparison": comparison,
    }
    summary_path = analysis_dir / "priority_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  ✓ Summary written: {summary_path}")

    print("\n" + "=" * 60)
    print("✓ STEP 9 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()