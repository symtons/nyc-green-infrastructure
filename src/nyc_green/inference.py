"""Step 7 — Sliding-window inference on the full NYC raster.

Takes the trained U-Net and applies it across the entire preprocessed Landsat
raster with overlapping tiles. Predictions in overlapping regions are averaged
(soft voting) to smooth out edge artifacts, then argmax'd to get final classes.

The output is a single GeoTIFF on the same grid as the aligned 30m rasters,
clipped to the NYC boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nyc_green.dataset import CHANNEL_MEAN, CHANNEL_STD
from nyc_green.tiles import compute_ndvi
from nyc_green.model import build_unet


def load_trained_model(
    checkpoint_path: Path,
    num_classes: int = 3,
    in_channels: int = 5,
    device: str = "cuda",
) -> torch.nn.Module:
    """Load the best U-Net checkpoint for inference."""
    model = build_unet(
        num_classes=num_classes,
        in_channels=in_channels,
        encoder_name="resnet18",
        encoder_weights=None,  # we're loading our own weights, skip the download
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def prepare_input_raster(landsat_path: Path) -> tuple[np.ndarray, dict]:
    """Load the aligned Landsat, compute NDVI, stack to 5 channels, normalize.

    Returns (image_stack, profile) where image_stack is (5, H, W) float32,
    normalized per-channel, and NaN-filled with channel means.
    """
    with rasterio.open(landsat_path) as src:
        landsat = src.read().astype(np.float32)  # (4, H, W)
        profile = src.profile.copy()

    # NDVI as 5th channel
    ndvi = compute_ndvi(landsat[2], landsat[3])
    stack = np.concatenate([landsat, ndvi[np.newaxis, ...]], axis=0)  # (5, H, W)

    # Fill NaNs with per-channel mean (same as training dataset does)
    for c in range(stack.shape[0]):
        nan_mask = np.isnan(stack[c])
        if nan_mask.any():
            stack[c][nan_mask] = CHANNEL_MEAN[c]

    # Normalize per channel
    stack = (stack - CHANNEL_MEAN[:, None, None]) / CHANNEL_STD[:, None, None]
    return stack.astype(np.float32), profile


def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
    num_classes: int = 3,
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Run tiled inference with overlap blending.

    Returns a (num_classes, H, W) float32 probability map, summed across
    overlapping tiles and normalized by coverage count.
    """
    _, H, W = image.shape

    # Pad the image so tiles tile cleanly at the edges
    pad_h = (tile_size - H % stride) % stride
    pad_w = (tile_size - W % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    _, Hp, Wp = image.shape

    # Accumulators for soft voting
    prob_sum = np.zeros((num_classes, Hp, Wp), dtype=np.float32)
    count = np.zeros((Hp, Wp), dtype=np.float32)

    # Enumerate tile positions
    positions = []
    for r in range(0, Hp - tile_size + 1, stride):
        for c in range(0, Wp - tile_size + 1, stride):
            positions.append((r, c))
    # Make sure we cover the very last row/column even if stride doesn't land exactly
    if positions[-1][0] + tile_size < Hp:
        for c in range(0, Wp - tile_size + 1, stride):
            positions.append((Hp - tile_size, c))
    if positions[-1][1] + tile_size < Wp:
        for r in range(0, Hp - tile_size + 1, stride):
            positions.append((r, Wp - tile_size))

    print(f"  Total tiles to infer: {len(positions)}")

    # Process in batches
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(positions), batch_size), desc="  infer"):
            batch_positions = positions[batch_start:batch_start + batch_size]
            tiles = np.stack([
                image[:, r:r + tile_size, c:c + tile_size]
                for (r, c) in batch_positions
            ])  # (B, 5, tile, tile)
            tiles_tensor = torch.from_numpy(tiles).to(device)

            logits = model(tiles_tensor)           # (B, num_classes, tile, tile)
            probs = F.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()

            for i, (r, c) in enumerate(batch_positions):
                prob_sum[:, r:r + tile_size, c:c + tile_size] += probs_np[i]
                count[r:r + tile_size, c:c + tile_size] += 1.0

    # Average probabilities across overlaps
    count = np.maximum(count, 1.0)
    prob_avg = prob_sum / count[None, :, :]

    # Crop back to original size
    prob_avg = prob_avg[:, :H, :W]
    return prob_avg


def probs_to_classes(prob_map: np.ndarray) -> np.ndarray:
    """Argmax the probability map to per-pixel class predictions (uint8)."""
    return prob_map.argmax(axis=0).astype(np.uint8)


def apply_boundary_mask(
    class_map: np.ndarray,
    reference_landsat_path: Path,
    nodata_value: int = 255,
) -> np.ndarray:
    """Set pixels to nodata wherever the reference Landsat is NaN.

    The aligned Landsat already has NaN outside the NYC boundary from Step 4,
    so we reuse that as the mask.
    """
    with rasterio.open(reference_landsat_path) as src:
        blue = src.read(1)
    out = class_map.copy()
    out[np.isnan(blue)] = nodata_value
    return out


def write_landcover_raster(
    class_map: np.ndarray,
    reference_profile: dict,
    out_path: Path,
):
    """Write the predicted land cover map as a single-band uint8 GeoTIFF."""
    profile = reference_profile.copy()
    profile.update({
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "lzw",
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(class_map, 1)


def compare_to_worldcover(
    predicted_path: Path,
    worldcover_path: Path,
    num_classes: int = 3,
) -> dict:
    """Compute pixelwise agreement and per-class IoU between two land cover rasters.

    Both rasters must share the same grid (which they will, since they come
    from the Step 4 alignment).
    """
    with rasterio.open(predicted_path) as src:
        pred = src.read(1)
    with rasterio.open(worldcover_path) as src:
        wc_raw = src.read(1)

    # Reclassify WorldCover to match our 3-class taxonomy
    from nyc_green.tiles import reclassify_worldcover
    wc_3class = reclassify_worldcover(wc_raw)

    # Valid pixels: both sources have a real class (not 255 nodata)
    valid = (pred != 255) & (wc_3class != 255)
    total = int(valid.sum())
    if total == 0:
        return {"agreement": 0.0, "iou_per_class": [0.0] * num_classes}

    agreement = float(((pred == wc_3class) & valid).sum() / total)

    # Per-class IoU (treating WorldCover as "ground truth")
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        true_c = (wc_3class == c) & valid
        inter = int((pred_c & true_c).sum())
        union = int((pred_c | true_c).sum())
        iou_per_class.append(inter / union if union > 0 else 0.0)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] = int(((wc_3class == t) & (pred == p) & valid).sum())

    return {
        "total_pixels": total,
        "agreement": agreement,
        "iou_per_class": iou_per_class,
        "confusion_matrix": cm.tolist(),
    }