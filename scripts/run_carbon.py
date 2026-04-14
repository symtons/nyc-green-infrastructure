"""Step 10.5 — Compute carbon sequestration summary for the dashboard."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.carbon import compute_carbon_summary


def main():
    print("=" * 60)
    print("STEP 10.5 — CARBON SEQUESTRATION ACCOUNTING")
    print("=" * 60)

    cfg = load_config()
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]

    landsat_path = interim_dir / "nyc_landsat_30m.tif"
    landcover_path = processed_dir / "nyc_landcover_predicted.tif"

    for p in (landsat_path, landcover_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required raster: {p}")

    print(f"\nSource rasters:")
    print(f"  Landsat:   {landsat_path.name}")
    print(f"  Landcover: {landcover_path.name}")

    print("\nComputing NDVI tiers and carbon totals...")
    summary = compute_carbon_summary(
        landsat_path=landsat_path,
        landcover_path=landcover_path,
    )

    # Print a human-readable summary
    print("\n" + "-" * 60)
    print("Per-tier breakdown:")
    print("-" * 60)
    for tier in summary["tiers"]:
        print(f"  {tier['name']:22s} "
              f"{tier['area_ha']:>10,.0f} ha  "
              f"@ {tier['rate_t_per_ha_yr']:>4.0f} tCO2/ha/yr  "
              f"= {tier['carbon_t_per_yr']:>10,.0f} tCO2/yr")

    totals = summary["totals"]
    print("-" * 60)
    print(f"  TOTAL vegetation area:   {totals['vegetation_area_ha']:,.0f} ha")
    print(f"  TOTAL carbon per year:   {totals['carbon_t_per_yr']:,.0f} tCO2/yr")
    print(f"  Area below sparse NDVI:  {totals['area_excluded_low_ndvi_ha']:,.0f} ha (rate 0)")

    print("\nReal-world equivalents:")
    eq = summary["equivalents"]
    print(f"  Equivalent to taking off the road:  {eq['cars_per_year']:>12,} cars/year")
    print(f"  Equivalent to growing for 10 years: {eq['tree_seedlings_10yr']:>12,} tree seedlings")
    print(f"  Equivalent to the energy used by:   {eq['homes_per_year']:>12,} homes/year")

    # Write to JSON for the dashboard
    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "carbon_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print("\n" + "=" * 60)
    print("Step 10.5 complete")
    print("=" * 60)


if __name__ == "__main__":
    main()