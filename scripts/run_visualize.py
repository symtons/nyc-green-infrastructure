"""Step 10 entry point — generate all visualizations and recommendations."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.viz_static import (
    plot_priority_zones,
    plot_continuous_raster,
    plot_landcover,
    plot_component_grid,
)
from nyc_green.viz_interactive import build_priority_map
from nyc_green.interventions import recommend_for_area, compute_area_ha_from_zones
from nyc_green.priority import PRIORITY_CRITICAL, PRIORITY_HIGH


def main():
    print("=" * 60)
    print("STEP 10 — VISUALIZATION & RECOMMENDATIONS")
    print("=" * 60)

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    maps_dir = outputs_dir / "maps"

    # ------------------------------------------------------------------
    # Static maps
    # ------------------------------------------------------------------
    print("\n[1/3] Generating static figures...")

    for label in ("worldcover", "model"):
        zones = maps_dir / f"priority_zones_{label}.tif"
        if not zones.exists():
            print(f"  ! missing {zones.name}, skip")
            continue
        out = figures_dir / f"priority_zones_{label}.png"
        plot_priority_zones(
            zones_path=zones,
            title=f"NYC Green Infrastructure Priority Zones\n(land cover: {label})",
            out_path=out,
            source_label=f"Source: {label}",
        )
        print(f"  OK  {out.relative_to(outputs_dir)}")

    lst_path = interim_dir / "nyc_lst_30m.tif"
    if lst_path.exists():
        plot_continuous_raster(
            raster_path=lst_path,
            title="NYC Land Surface Temperature (Summer 2024)",
            out_path=figures_dir / "context_lst.png",
            cmap="hot",
            label="Degrees C",
        )
        print(f"  OK  context_lst.png")

    eq_path = interim_dir / "nyc_equity_30m.tif"
    if eq_path.exists():
        plot_continuous_raster(
            raster_path=eq_path,
            title="NYC Heat Vulnerability Index (scaled 0-100)",
            out_path=figures_dir / "context_equity.png",
            cmap="viridis",
            label="HVI (0 = lowest risk, 100 = highest)",
            vmin=0, vmax=100,
        )
        print(f"  OK  context_equity.png")

    lc_model = processed_dir / "nyc_landcover_predicted.tif"
    if lc_model.exists():
        plot_landcover(
            landcover_path=lc_model,
            title="NYC Land Cover — U-Net predictions",
            out_path=figures_dir / "landcover_model.png",
            source="model",
        )
        print(f"  OK  landcover_model.png")

    lc_wc = interim_dir / "nyc_worldcover_30m.tif"
    if lc_wc.exists():
        plot_landcover(
            landcover_path=lc_wc,
            title="NYC Land Cover — ESA WorldCover (reclassified)",
            out_path=figures_dir / "landcover_worldcover.png",
            source="worldcover",
        )
        print(f"  OK  landcover_worldcover.png")

    # ------------------------------------------------------------------
    # Interactive Folium map
    # ------------------------------------------------------------------
    print("\n[2/3] Building interactive Folium map...")
    print("      (image overlays + Critical-zone marker cluster)")

    zones_paths = {
        "WorldCover": maps_dir / "priority_zones_worldcover.tif",
        "Model":      maps_dir / "priority_zones_model.tif",
    }
    zones_paths = {k: v for k, v in zones_paths.items() if v.exists()}

    if zones_paths:
        interactive_out = maps_dir / "priority_zones_interactive.html"

        lst_raster = interim_dir / "nyc_lst_30m.tif"
        landsat_raster = interim_dir / "nyc_landsat_30m.tif"
        equity_raster = interim_dir / "nyc_equity_30m.tif"
        score_raster = maps_dir / "priority_score_model.tif"

        borough_geojson = interim_dir / "nyc_boundary.geojson"
        modzcta_geojson = raw_dir / "equity" / "nyc_hvi_joined.geojson"

        build_priority_map(
            zones_paths=zones_paths,
            out_path=interactive_out,
            lst_raster_path=lst_raster if lst_raster.exists() else None,
            landsat_raster_path=landsat_raster if landsat_raster.exists() else None,
            equity_raster_path=equity_raster if equity_raster.exists() else None,
            score_raster_path=score_raster if score_raster.exists() else None,
            borough_geojson=borough_geojson if borough_geojson.exists() else None,
            modzcta_geojson=modzcta_geojson if modzcta_geojson.exists() else None,
        )
        print(f"  OK  {interactive_out.relative_to(outputs_dir)}")
    else:
        print("  !   no priority zones rasters found, skip")

    # ------------------------------------------------------------------
    # Intervention recommendations (kept for completeness — not shown
    # in the dashboard, but the CSV exists on disk if needed later)
    # ------------------------------------------------------------------
    print("\n[3/3] Computing intervention recommendations (background tables)...")

    all_recommendations = {}
    for label in ("worldcover", "model"):
        zones = maps_dir / f"priority_zones_{label}.tif"
        if not zones.exists():
            continue
        critical_ha = compute_area_ha_from_zones(zones, PRIORITY_CRITICAL)
        high_ha = compute_area_ha_from_zones(zones, PRIORITY_HIGH)

        print(f"\n  [{label}]")
        print(f"    Critical area: {critical_ha:,.1f} ha")
        print(f"    High area:     {high_ha:,.1f} ha")

        df = recommend_for_area(critical_ha, high_ha)
        csv_path = tables_dir / f"interventions_{label}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"    OK  {csv_path.relative_to(outputs_dir)}")

        total_row = df[df["intervention"] == "TOTAL"].iloc[0]
        all_recommendations[label] = {
            "critical_ha": critical_ha,
            "high_ha": high_ha,
            "portfolio_csv": str(csv_path),
            "total_carbon_t_yr": float(total_row["carbon_t_per_year"]),
        }

    analysis_dir = outputs_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "interventions_summary.json", "w") as f:
        json.dump(all_recommendations, f, indent=2)

    print("\n" + "=" * 60)
    print("STEP 10 COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  figures:  {figures_dir}")
    print(f"  tables:   {tables_dir}")
    print(f"  maps:     {maps_dir}")
    print(f"  analysis: {analysis_dir}")


if __name__ == "__main__":
    main()