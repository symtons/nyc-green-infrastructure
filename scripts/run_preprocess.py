"""Step 4 entry point — run preprocessing end-to-end.

Produces NYC-wide aligned rasters on a common 30m UTM grid.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_green.config import load_config
from nyc_green.preprocessing import (
    TARGET_CRS,
    TARGET_RES_M,
    merge_rasters,
    reproject_to_utm,
    align_to_reference,
    load_borough_union,
    clip_to_polygon,
    write_raster,
)


BOROUGHS = ["manhattan", "brooklyn", "bronx", "queens", "staten_island"]


def main():
    print("=" * 60)
    print("STEP 4 — PREPROCESSING & ALIGNMENT")
    print(f"Target CRS: {TARGET_CRS}  |  Resolution: {TARGET_RES_M}m")
    print("=" * 60)

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_dir"]
    interim_dir = cfg["paths"]["interim_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    boundaries_path = raw_dir / "boundaries" / "nyc_boroughs.geojson"

    # --- 1. Load borough union in target CRS (needed for clipping later)
    print("\n[1/5] Loading and unioning borough boundaries...")
    union_gdf = load_borough_union(boundaries_path, TARGET_CRS)
    union_out = interim_dir / "nyc_boundary.geojson"
    union_gdf.to_file(union_out, driver="GeoJSON")
    print(f"  ✓ Union polygon saved: {union_out.name}")
    print(f"  Area: {union_gdf.geometry.area.sum() / 1e6:.1f} km²")

    # --- 2. Merge per-borough Landsat, reproject to UTM 30m, clip
    # --- 2. Reproject each Landsat borough to UTM, then mosaic in UTM space.
    #       We go reproject-then-merge (instead of merge-then-reproject) because
    #       the GEE-exported TIFFs have inconsistent TIFF headers (SamplesPerPixel
    #       vs indexes) that confuse rasterio.merge. Reprojecting each file
    #       individually sidesteps the issue, and the final result is identical.
    print("\n[2/5] Reprojecting per-borough Landsat to UTM 18N @ 30m...")
    landsat_paths = [raw_dir / "landsat" / f"{b}_landsat.tif" for b in BOROUGHS]
    missing = [p for p in landsat_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Landsat files: {missing}")

    # First reproject each borough to UTM individually, write to temp files
    temp_utm_files = []
    for p in landsat_paths:
        print(f"  Reprojecting {p.name}...")
        with rasterio.open(p) as src:
            # Explicitly read exactly 4 bands (blue, green, red, nir)
            src_array = src.read(indexes=[1, 2, 3, 4])
            src_crs = src.crs
            src_transform = src.transform
            src_nodata = src.nodata

        utm_array, utm_tfm, profile_updates = reproject_to_utm(
            src_array,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            resampling=Resampling.bilinear,
        )

        temp_path = interim_dir / f"_temp_{p.stem}_utm.tif"
        temp_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 4,
            "crs": profile_updates["crs"],
            "transform": profile_updates["transform"],
            "width": profile_updates["width"],
            "height": profile_updates["height"],
            "nodata": np.nan,
            "compress": "lzw",
        }
        write_raster(utm_array.astype(np.float32), temp_profile, temp_path)
        temp_utm_files.append(temp_path)

    # Now merge the UTM-reprojected files. They share the same CRS and resolution,
    # so this merge is straightforward.
    print(f"  Merging {len(temp_utm_files)} UTM rasters into NYC mosaic...")
    srcs = [rasterio.open(p) for p in temp_utm_files]
    try:
        mosaic, mosaic_tfm = merge(srcs, indexes=[1, 2, 3, 4])
    finally:
        for s in srcs:
            s.close()

    merged_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 4,
        "crs": TARGET_CRS,
        "transform": mosaic_tfm,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "nodata": np.nan,
        "compress": "lzw",
    }
    print(f"  Mosaic shape: {mosaic.shape}")

    print("  Clipping to NYC boundary...")
    clipped, clip_tfm, clip_profile = clip_to_polygon(
        mosaic.astype(np.float32), merged_profile, union_gdf
    )
    landsat_out = interim_dir / "nyc_landsat_30m.tif"
    write_raster(clipped, clip_profile, landsat_out)
    print(f"  ✓ Saved: {landsat_out.name}  shape={clipped.shape}")

    # Clean up temp files
    for p in temp_utm_files:
        p.unlink()

    # Save the reference profile — everything else aligns to this
    reference_profile = clip_profile

    # --- 3. LST: reproject NYC-wide file to UTM, align to reference grid, clip
    print("\n[3/5] Processing NYC LST...")
    lst_path = raw_dir / "landsat" / "nyc_lst.tif"
    if not lst_path.exists():
        raise FileNotFoundError(f"Missing LST file: {lst_path}")

    lst_aligned = align_to_reference(lst_path, reference_profile, Resampling.bilinear)
    lst_profile = reference_profile.copy()
    lst_profile.update({"count": 1, "dtype": "float32"})
    lst_clipped, _, lst_clip_profile = clip_to_polygon(
        lst_aligned.astype(np.float32), lst_profile, union_gdf
    )
    lst_out = interim_dir / "nyc_lst_30m.tif"
    write_raster(lst_clipped, lst_clip_profile, lst_out)
    print(f"  ✓ Saved: {lst_out.name}  shape={lst_clipped.shape}")

    # --- 4. WorldCover: merge per-borough, align to reference (nearest neighbor)
    print("\n[4/5] Merging per-borough WorldCover...")
    wc_paths = [raw_dir / "landcover" / f"{b}_worldcover.tif" for b in BOROUGHS]
    missing = [p for p in wc_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing WorldCover files: {missing}")

    wc_mosaic, wc_tfm, wc_profile = merge_rasters(wc_paths)
    print(f"  Merged WorldCover shape: {wc_mosaic.shape}")

    # Write merged mosaic to a temp file so align_to_reference can use it
    temp_wc = interim_dir / "_temp_wc_merged.tif"
    wc_profile_merged = wc_profile.copy()
    wc_profile_merged.update({
        "height": wc_mosaic.shape[1],
        "width": wc_mosaic.shape[2],
        "transform": wc_tfm,
        "driver": "GTiff",
        "compress": "lzw",
    })
    write_raster(wc_mosaic, wc_profile_merged, temp_wc)

    print("  Aligning to reference grid (nearest neighbor)...")
    wc_aligned = align_to_reference(temp_wc, reference_profile, Resampling.nearest)
    wc_out_profile = reference_profile.copy()
    wc_out_profile.update({"count": 1, "dtype": "uint8", "nodata": 0})
    wc_clipped, _, wc_clip_profile = clip_to_polygon(
        wc_aligned.astype(np.uint8), wc_out_profile, union_gdf
    )
    wc_out = interim_dir / "nyc_worldcover_30m.tif"
    write_raster(wc_clipped, wc_clip_profile, wc_out)
    print(f"  ✓ Saved: {wc_out.name}  shape={wc_clipped.shape}")

    # Clean up temp file
    temp_wc.unlink()

    # --- 5. Verification: all three rasters should share the same grid
    print("\n[5/5] Verifying grid alignment...")
    outputs = {
        "Landsat":    landsat_out,
        "LST":        lst_out,
        "WorldCover": wc_out,
    }

    shapes = {}
    transforms = {}
    crs_set = set()
    for name, path in outputs.items():
        with rasterio.open(path) as src:
            shapes[name] = (src.height, src.width)
            transforms[name] = src.transform
            crs_set.add(str(src.crs))

    print(f"  CRS (all files): {crs_set}")
    for name, shape in shapes.items():
        print(f"  {name:11s} shape={shape}")

    assert len(crs_set) == 1, f"CRS mismatch: {crs_set}"
    assert len(set(shapes.values())) == 1, f"Shape mismatch: {shapes}"
    print("  ✓ All rasters share the same grid.")

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Outputs in: {interim_dir}")
    for name, path in outputs.items():
        size_mb = path.stat().st_size / 1_000_000
        print(f"    {path.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()