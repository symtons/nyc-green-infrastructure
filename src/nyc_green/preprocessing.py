"""Step 4 — Preprocessing: merge, reproject, align, and clip rasters.

Takes per-borough raw rasters and produces NYC-wide aligned rasters on a
common grid in UTM Zone 18N (EPSG:32618) at 30m resolution.

Outputs go to data/interim/:
    nyc_landsat_30m.tif     — 4-band (blue, green, red, nir), float32
    nyc_lst_30m.tif         — 1-band LST in °C, float32
    nyc_worldcover_30m.tif  — 1-band land cover, uint8 (WorldCover raw classes)
    nyc_boundary.geojson    — union of 5 borough polygons in target CRS
"""
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from shapely.ops import unary_union


TARGET_CRS = "EPSG:32618"   # UTM Zone 18N (meters)
TARGET_RES_M = 30            # 30 meter pixels


# ==========================================================================
# Merging & reprojection primitives
# ==========================================================================

def merge_rasters(input_paths: List[Path], nodata_value=None) -> tuple:
    """Merge a list of rasters into a single mosaic.

    Returns (mosaic_array, mosaic_transform, source_profile).
    Assumes all inputs share the same CRS and band count.

    Explicitly passes `indexes=[1..N]` because GEE-exported TIFFs sometimes
    have inconsistent photometric headers that confuse rasterio.merge's
    band auto-detection (causing DatasetIOShapeError).
    """
    srcs = [rasterio.open(p) for p in input_paths]
    try:
        n_bands = srcs[0].count
        band_indexes = list(range(1, n_bands + 1))  # rasterio is 1-indexed
        mosaic, mosaic_transform = merge(
            srcs,
            indexes=band_indexes,
            nodata=nodata_value,
        )
        profile = srcs[0].profile.copy()
        profile["count"] = n_bands
    finally:
        for s in srcs:
            s.close()
    return mosaic, mosaic_transform, profile

def reproject_to_utm(
    src_array: np.ndarray,
    src_transform,
    src_crs,
    src_nodata,
    resampling: Resampling,
    target_crs: str = TARGET_CRS,
    target_res_m: int = TARGET_RES_M,
) -> tuple:
    """Reproject a raster array to the target CRS and resolution.

    Returns (dst_array, dst_transform, dst_profile_updates).
    """
    n_bands = src_array.shape[0] if src_array.ndim == 3 else 1
    if src_array.ndim == 2:
        src_array = src_array[np.newaxis, :, :]

    # Compute the bounds of the source in its native CRS
    src_height, src_width = src_array.shape[1], src_array.shape[2]
    left, top = src_transform * (0, 0)
    right, bottom = src_transform * (src_width, src_height)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        target_crs,
        src_width,
        src_height,
        left=left, bottom=bottom, right=right, top=top,
        resolution=target_res_m,
    )

    dst_array = np.full(
        (n_bands, dst_height, dst_width),
        fill_value=np.nan if np.issubdtype(src_array.dtype, np.floating) else 0,
        dtype=src_array.dtype,
    )

    for b in range(n_bands):
        reproject(
            source=src_array[b],
            destination=dst_array[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
        )

    return dst_array, dst_transform, {
        "crs": target_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
    }


# ==========================================================================
# Alignment to a reference grid
# ==========================================================================

def align_to_reference(
    src_path: Path,
    ref_profile: dict,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject and resample a source raster to exactly match a reference grid.

    The result has identical shape, transform, and CRS to the reference.
    """
    with rasterio.open(src_path) as src:
        dst_array = np.full(
            (src.count, ref_profile["height"], ref_profile["width"]),
            fill_value=np.nan if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else 0,
            dtype=src.dtypes[0],
        )
        for b in range(src.count):
            reproject(
                source=rasterio.band(src, b + 1),
                destination=dst_array[b],
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                dst_width=ref_profile["width"],
                dst_height=ref_profile["height"],
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
    return dst_array


# ==========================================================================
# Clipping to borough polygons
# ==========================================================================

def load_borough_union(boundaries_path: Path, target_crs: str = TARGET_CRS):
    """Load all 5 borough polygons, union them, reproject to target CRS.

    Returns a GeoDataFrame with one row: the union polygon.
    """
    gdf = gpd.read_file(boundaries_path)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(target_crs)
    union_geom = unary_union(gdf.geometry.values)
    return gpd.GeoDataFrame({"geometry": [union_geom]}, crs=target_crs)


def clip_to_polygon(
    array: np.ndarray,
    profile: dict,
    polygon_gdf: gpd.GeoDataFrame,
) -> tuple:
    """Clip a raster array to a polygon, setting outside pixels to nodata.

    Returns (clipped_array, clipped_transform, clipped_profile).
    """
    # Write to in-memory file so rio_mask can use it
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(array)
        with memfile.open() as dataset:
            clipped, clip_transform = rio_mask(
                dataset,
                polygon_gdf.geometry.values,
                crop=True,
                nodata=profile.get("nodata"),
            )

    new_profile = profile.copy()
    new_profile.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": clip_transform,
    })
    return clipped, clip_transform, new_profile


# ==========================================================================
# Writers
# ==========================================================================

def write_raster(array: np.ndarray, profile: dict, out_path: Path):
    """Write an array to a GeoTIFF with the given profile."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": array.shape[0] if array.ndim == 3 else 1,
        "height": array.shape[-2],
        "width": array.shape[-1],
        "compress": "lzw",
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)