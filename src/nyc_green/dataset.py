"""Step 6 — PyTorch Dataset for tile-based land cover segmentation.

Reads tiles produced by Step 5 from data/processed/tiles/ and serves them
to the training loop with normalization and optional augmentation.

The dataset honors the train/val/test split frozen in tile_metadata.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# Normalization stats computed empirically from a sample of NYC Landsat tiles.
# For ImageNet-pretrained ResNet, the first 3 channels (BGR) should be
# normalized approximately to the ImageNet statistics. NIR and NDVI get
# their own sensible ranges.
# These are reasonable defaults — you can recompute them on your own data
# later if you want to be precise.
CHANNEL_MEAN = np.array([0.08, 0.09, 0.10, 0.20, 0.20], dtype=np.float32)
CHANNEL_STD  = np.array([0.06, 0.06, 0.07, 0.10, 0.25], dtype=np.float32)


class TileDataset(Dataset):
    """Loads 5-channel image tiles and 3-class masks from disk."""

    def __init__(
        self,
        tiles_dir: Path,
        split: str,
        augment: bool = False,
    ):
        self.tiles_dir = Path(tiles_dir)
        self.images_dir = self.tiles_dir / "images"
        self.masks_dir = self.tiles_dir / "masks"
        self.split = split

        # Load the metadata CSV and filter to this split
        meta_path = self.tiles_dir / "tile_metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Run Step 5 first.")
        df = pd.read_csv(meta_path)
        self.metadata = df[df.split == split].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(f"No tiles in split '{split}'")

        # Augmentation pipeline (spatial only — we normalize channels manually)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=0.05,
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.3,
                ),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        tile_id = row["tile_id"]

        # Load image (5, 256, 256) float32 and mask (256, 256) uint8
        image = np.load(self.images_dir / f"{tile_id}.npy")
        mask = np.load(self.masks_dir / f"{tile_id}.npy")

        # Replace NaN (nodata from clipping) with per-channel mean so the
        # normalized value becomes ~0 and the model treats it as neutral
        for c in range(image.shape[0]):
            channel = image[c]
            nan_mask = np.isnan(channel)
            if nan_mask.any():
                channel[nan_mask] = CHANNEL_MEAN[c]
                image[c] = channel

        # Normalize per channel
        image = (image - CHANNEL_MEAN[:, None, None]) / CHANNEL_STD[:, None, None]
        image = image.astype(np.float32)

        # Mask nodata (255) -> 0 (vegetation) is wrong; better: set to an
        # ignore_index that the loss function skips. We use 255 as ignore.
        # (The loss will be constructed with ignore_index=255.)

        # Apply augmentation — albumentations expects HWC format for images
        if self.transform is not None:
            # (C, H, W) -> (H, W, C) for albumentations
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented["image"], (2, 0, 1))
            mask = augmented["mask"]

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return image_tensor, mask_tensor


def make_balanced_sampler(
    dataset: TileDataset,
    rare_class_col: str = "water_frac",
    boost: float = 5.0,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples tiles containing the rare class.

    For each tile, weight = 1 + boost * (fraction of rare class in that tile).
    Tiles with 0% rare class get weight 1.0; tiles with lots of rare class
    get up to (1 + boost) = 6.0. This biases the training loader toward
    showing the model more of the minority class without making it dominant.
    """
    fractions = dataset.metadata[rare_class_col].values
    weights = 1.0 + boost * fractions
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(dataset),
        replacement=True,
    )