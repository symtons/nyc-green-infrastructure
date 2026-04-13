"""Step 5 — Tile generation for U-Net training.

Reads the aligned 30m rasters from data/interim/ and produces:
  - 5-channel image tiles (blue, green, red, nir, ndvi), float32
  - 1-channel mask tiles (3-class: vegetation, water, built-up), uint8
  - Per-tile class statistics for later balanced sampling
  - A frozen train/val/test split

Output layout:
    data/processed/tiles/
        images/tile_0000.npy   (5, 256, 256) float32
        masks/tile_0000.npy    (256, 256)    uint8
        tile_metadata.csv      per-tile class stats + split assignment
        split_info.json        summary of the split
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio


# WorldCover v200 -> 3-class simplified taxonomy
#   0 = vegetation, 1 = water, 2 = built-up
# Bare/open was dropped (under 1% of NYC land, cannot be learned reliably).
# Former bare/open codes (cropland, bare, snow, moss) now map to built-up
# because in NYC they visually and functionally resemble impervious surface
# (gravel lots, construction sites, stadium fields, rooftop gardens).
WORLDCOVER_TO_3CLASS = {
    10:  0,  # Tree cover         -> vegetation
    20:  0,  # Shrubland          -> vegetation
    30:  0,  # Grassland          -> vegetation
    40:  2,  # Cropland           -> built-up (was bare/open in v1)
    50:  2,  # Built-up           -> built-up
    60:  2,  # Bare/sparse veg    -> built-up (was bare/open in v1)
    70:  2,  # Snow and ice       -> built-up (extremely rare in summer NYC)
    80:  1,  # Permanent water    -> water
    90:  1,  # Herbaceous wetland -> water
    95:  1,  # Mangroves          -> water
    100: 2,  # Moss and lichen    -> built-up
}
NUM_CLASSES = 3
CLASS_NAMES = ["vegetation", "water", "built_up"]


@dataclass
class TileMetadata:
    tile_id: str
    row: int
    col: int
    veg_frac: float
    water_frac: float
    built_frac: float
    valid_frac: float
    split: str  # 'train' | 'val' | 'test'


# ==========================================================================
# Core operations
# ==========================================================================

def compute_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red)."""
    denom = nir + red + eps
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def reclassify_worldcover(wc: np.ndarray) -> np.ndarray:
    """Map WorldCover codes to 3 simplified classes. Unknown -> 255 (nodata)."""
    out = np.full_like(wc, 255, dtype=np.uint8)
    for wc_code, our_code in WORLDCOVER_TO_3CLASS.items():
        out[wc == wc_code] = our_code
    return out


def is_tile_valid(
    image_tile: np.ndarray,
    mask_tile: np.ndarray,
    min_valid_frac: float = 0.50,
) -> tuple[bool, float]:
    """A tile is valid if enough pixels are not NaN (image) and not 255 (mask)."""
    image_valid = ~np.isnan(image_tile[0])
    mask_valid = mask_tile != 255
    both_valid = image_valid & mask_valid
    valid_frac = both_valid.mean()
    return valid_frac >= min_valid_frac, float(valid_frac)


def class_fractions(mask_tile: np.ndarray) -> dict:
    """Fraction of each class in a mask tile (ignoring nodata 255)."""
    valid = mask_tile != 255
    total = max(int(valid.sum()), 1)
    return {
        "veg_frac":   float(((mask_tile == 0) & valid).sum() / total),
        "water_frac": float(((mask_tile == 1) & valid).sum() / total),
        "built_frac": float(((mask_tile == 2) & valid).sum() / total),
    }


# ==========================================================================
# Tile generation pipeline
# ==========================================================================

def generate_tiles(
    landsat_path: Path,
    worldcover_path: Path,
    out_dir: Path,
    tile_size: int = 256,
    stride: int = 128,
    min_valid_frac: float = 0.50,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Slice aligned rasters into training tiles and write them to disk."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading aligned rasters into memory...")
    with rasterio.open(landsat_path) as src:
        landsat = src.read().astype(np.float32)        # (4, H, W)
        landsat_nodata = src.nodata
        H, W = src.height, src.width
    with rasterio.open(worldcover_path) as src:
        wc_raw = src.read(1).astype(np.uint8)

    assert landsat.shape[1:] == wc_raw.shape, (
        f"Shape mismatch: landsat {landsat.shape[1:]} vs worldcover {wc_raw.shape}"
    )

    print(f"  Raster shape: {H} x {W}")
    print("  Computing NDVI...")
    ndvi = compute_ndvi(landsat[2], landsat[3])

    image_stack = np.concatenate([landsat, ndvi[np.newaxis, ...]], axis=0)

    if landsat_nodata is not None and not np.isnan(landsat_nodata):
        image_stack[image_stack == landsat_nodata] = np.nan

    print("  Reclassifying WorldCover to 3 classes...")
    mask = reclassify_worldcover(wc_raw)

    print(f"  Slicing into {tile_size}x{tile_size} tiles, stride={stride}...")
    tile_rows = list(range(0, H - tile_size + 1, stride))
    tile_cols = list(range(0, W - tile_size + 1, stride))
    total_positions = len(tile_rows) * len(tile_cols)
    print(f"  Candidate positions: {total_positions}")

    metadata: List[TileMetadata] = []
    tile_count = 0
    kept = 0

    for r in tile_rows:
        for c in tile_cols:
            image_tile = image_stack[:, r:r + tile_size, c:c + tile_size]
            mask_tile = mask[r:r + tile_size, c:c + tile_size]

            valid, valid_frac = is_tile_valid(image_tile, mask_tile, min_valid_frac)
            tile_count += 1
            if not valid:
                continue

            tile_id = f"tile_{kept:04d}"
            np.save(images_dir / f"{tile_id}.npy", image_tile.astype(np.float32))
            np.save(masks_dir / f"{tile_id}.npy", mask_tile.astype(np.uint8))

            fracs = class_fractions(mask_tile)
            metadata.append(TileMetadata(
                tile_id=tile_id,
                row=r,
                col=c,
                valid_frac=valid_frac,
                split="",
                **fracs,
            ))
            kept += 1

    print(f"  Kept {kept}/{tile_count} tiles ({kept/tile_count*100:.1f}%)")
    if kept == 0:
        raise RuntimeError("No valid tiles produced — check raster alignment and validity threshold")

    # Train/val/test split
    rng = np.random.default_rng(random_seed)
    indices = np.arange(kept)
    rng.shuffle(indices)

    n_test = int(round(kept * test_frac))
    n_val  = int(round(kept * val_frac))
    n_train = kept - n_val - n_test

    split_labels = np.array(["train"] * kept, dtype=object)
    split_labels[indices[n_train:n_train + n_val]] = "val"
    split_labels[indices[n_train + n_val:]] = "test"

    for i, m in enumerate(metadata):
        m.split = split_labels[i]

    df = pd.DataFrame([asdict(m) for m in metadata])

    meta_csv = out_dir / "tile_metadata.csv"
    df.to_csv(meta_csv, index=False)

    split_info = {
        "total_tiles": int(kept),
        "train": int((df.split == "train").sum()),
        "val":   int((df.split == "val").sum()),
        "test":  int((df.split == "test").sum()),
        "random_seed": random_seed,
        "tile_size": tile_size,
        "stride": stride,
        "min_valid_frac": min_valid_frac,
        "per_split_class_means": {
            split: {
                cls: float(df.loc[df.split == split, f"{cls}_frac"].mean())
                for cls in ["veg", "water", "built"]
            }
            for split in ["train", "val", "test"]
        },
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return df