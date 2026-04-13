"""Step 5 entry point — generate training tiles from aligned rasters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.tiles import generate_tiles


def main():
    print("=" * 60)
    print("STEP 5 — TILE GENERATION")
    print("=" * 60)

    cfg = load_config()
    interim_dir = cfg["paths"]["interim_dir"]
    processed_dir = cfg["paths"]["processed_dir"]

    landsat_path = interim_dir / "nyc_landsat_30m.tif"
    worldcover_path = interim_dir / "nyc_worldcover_30m.tif"

    for p in (landsat_path, worldcover_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}. Run Step 4 first.")

    tiles_dir = processed_dir / "tiles"
    print(f"Output dir: {tiles_dir}\n")

    model_cfg = cfg["model"]
    df = generate_tiles(
        landsat_path=landsat_path,
        worldcover_path=worldcover_path,
        out_dir=tiles_dir,
        tile_size=model_cfg["tile_size"],
        stride=model_cfg["tile_size"] // 4,     # 50% overlap
        min_valid_frac=0.50,                    # coastal city -> lenient
        val_frac=model_cfg["val_split"],
        test_frac=model_cfg["test_split"],
        random_seed=model_cfg["random_seed"],
    )

    print("\n" + "=" * 60)
    print("TILE GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total tiles: {len(df)}")
    print(f"  train: {(df.split == 'train').sum()}")
    print(f"  val:   {(df.split == 'val').sum()}")
    print(f"  test:  {(df.split == 'test').sum()}")
    print()
    print("Mean class fractions (all tiles):")
    for col, name in [
        ("veg_frac",   "Vegetation"),
        ("water_frac", "Water     "),
        ("built_frac", "Built-up  "),
    ]:
        print(f"  {name}: {df[col].mean()*100:5.2f}%")
    print()
    print("Water tile counts by threshold (diagnostic for class imbalance):")
    for thresh in (0.0, 0.01, 0.05, 0.10):
        n = (df.water_frac > thresh).sum()
        print(f"  tiles with >{thresh*100:>4.1f}% water: {n}")
    print()
    print("Vegetation tile counts by threshold:")
    for thresh in (0.05, 0.10, 0.25, 0.50):
        n = (df.veg_frac > thresh).sum()
        print(f"  tiles with >{thresh*100:>4.1f}% vegetation: {n}")
    print()
    print("✓ Done. Metadata: tile_metadata.csv, split_info.json")


if __name__ == "__main__":
    main()