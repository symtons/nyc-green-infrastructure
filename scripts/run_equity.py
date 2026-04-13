"""Step 8 entry point — download HVI + MODZCTA, rasterize to 30m grid."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.equity import download_hvi_joined, rasterize_hvi


def main():
    print("=" * 60)
    print("STEP 8 — EQUITY LAYER (NYC HEAT VULNERABILITY INDEX)")
    print("=" * 60)

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    interim_dir = cfg["paths"]["interim_dir"]

    hvi_geojson_path = raw_dir / "equity" / "nyc_hvi_joined.geojson"
    reference_raster = interim_dir / "nyc_landsat_30m.tif"
    out_raster = interim_dir / "nyc_equity_30m.tif"

    if not reference_raster.exists():
        raise FileNotFoundError(f"Missing reference raster: {reference_raster}")

    # --- 1. Download and join ---
    print("\n[1/2] Downloading HVI and joining to MODZCTA polygons...")
    gdf = download_hvi_joined(hvi_geojson_path)
    print(f"  ✓ Joined features: {len(gdf)}")
    print(f"  Features with HVI score: {int(gdf['hvi'].notna().sum())}")
    vals = pd.to_numeric(gdf["hvi"], errors="coerce").dropna()
    if len(vals) > 0:
        print(f"  HVI raw range: min={vals.min()}  max={vals.max()}  mean={vals.mean():.2f}")

    # --- 2. Rasterize ---
    print("\n[2/2] Rasterizing to 30m UTM grid...")
    stats = rasterize_hvi(
        gdf,
        reference_raster_path=reference_raster,
        out_path=out_raster,
    )
    print(f"  ✓ Saved: {out_raster.name}")
    print(f"  Valid pixels: {stats['valid_pixels']:,}")
    print(
        f"  Scaled score range (0-100): "
        f"min={stats['score_min']:.1f}, "
        f"max={stats['score_max']:.1f}, "
        f"mean={stats['score_mean']:.1f}"
    )

    meta_path = interim_dir / "nyc_equity_30m_meta.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Metadata: {meta_path.name}")

    print("\n" + "=" * 60)
    print("✓ STEP 8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import pandas as pd  # used by main() above
    main()