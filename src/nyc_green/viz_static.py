"""Step 10 — Static maps for reports and the README.

Produces publication-quality PNGs from the aligned rasters. One figure per
priority map, plus context figures (LST, HVI, NDVI, land cover). Designed
to drop straight into a portfolio or a PDF brief.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm


# --- Colormaps ------------------------------------------------------------

PRIORITY_COLORS = [
    "#f5f5f5",  # 0 None/Excluded (light grey)
    "#ffe8a1",  # 1 Low           (pale yellow)
    "#ffb35c",  # 2 Moderate      (orange)
    "#ef5350",  # 3 High          (red)
    "#7f0000",  # 4 Critical      (dark red)
]
PRIORITY_CMAP = ListedColormap(PRIORITY_COLORS)
PRIORITY_LABELS = ["None", "Low", "Moderate", "High", "Critical"]

LANDCOVER_COLORS_3 = [
    "#2e7d32",  # 0 Vegetation (dark green)
    "#1976d2",  # 1 Water      (blue)
    "#9e9e9e",  # 2 Built-up   (grey)
]
LANDCOVER_CMAP_3 = ListedColormap(LANDCOVER_COLORS_3)
LANDCOVER_LABELS_3 = ["Vegetation", "Water", "Built-up"]


def _read_raster(path: Path) -> tuple[np.ndarray, tuple]:
    """Read a single-band raster and return (data, extent) for imshow."""
    with rasterio.open(path) as src:
        data = src.read(1)
        b = src.bounds
    extent = (b.left, b.right, b.bottom, b.top)
    return data, extent


def plot_priority_zones(
    zones_path: Path,
    title: str,
    out_path: Path,
    source_label: str = "",
):
    """Plot a priority zones raster with a categorical legend."""
    data, extent = _read_raster(zones_path)

    # Mask nodata (255) so it shows transparent
    masked = np.ma.masked_where(data == 255, data)

    fig, ax = plt.subplots(figsize=(10, 10))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], PRIORITY_CMAP.N)
    ax.imshow(masked, cmap=PRIORITY_CMAP, norm=norm, extent=extent, interpolation="nearest")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 18N)", fontsize=10)
    ax.set_ylabel("Northing (m, UTM 18N)", fontsize=10)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=label)
        for c, label in zip(PRIORITY_COLORS, PRIORITY_LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        title="Priority",
        title_fontsize=10,
    )

    # Source annotation
    if source_label:
        ax.text(
            0.01, 0.99, source_label,
            transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85, pad=4),
        )

    ax.set_aspect("equal")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_continuous_raster(
    raster_path: Path,
    title: str,
    out_path: Path,
    cmap: str = "magma",
    label: str = "",
    vmin: float = None,
    vmax: float = None,
):
    """Plot a continuous raster (LST, equity, NDVI...) with a colorbar."""
    data, extent = _read_raster(raster_path)
    valid = ~np.isnan(data)
    if not valid.any():
        return

    masked = np.ma.masked_invalid(data)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        masked, cmap=cmap, extent=extent,
        vmin=vmin if vmin is not None else float(np.nanpercentile(data[valid], 1)),
        vmax=vmax if vmax is not None else float(np.nanpercentile(data[valid], 99)),
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 18N)", fontsize=10)
    ax.set_ylabel("Northing (m, UTM 18N)", fontsize=10)
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_landcover(
    landcover_path: Path,
    title: str,
    out_path: Path,
    source: str = "model",
):
    """Plot a 3-class land cover map with a categorical legend.

    source: 'model' for our 3-class taxonomy, 'worldcover' for raw 11-class
            (which gets reclassified on the fly).
    """
    data, extent = _read_raster(landcover_path)

    if source == "worldcover":
        from nyc_green.tiles import reclassify_worldcover
        data = reclassify_worldcover(data)

    masked = np.ma.masked_where(data == 255, data)

    fig, ax = plt.subplots(figsize=(10, 10))
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], LANDCOVER_CMAP_3.N)
    ax.imshow(masked, cmap=LANDCOVER_CMAP_3, norm=norm, extent=extent, interpolation="nearest")

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Easting (m, UTM 18N)", fontsize=10)
    ax.set_ylabel("Northing (m, UTM 18N)", fontsize=10)
    ax.set_aspect("equal")

    patches = [
        mpatches.Patch(color=c, label=label)
        for c, label in zip(LANDCOVER_COLORS_3, LANDCOVER_LABELS_3)
    ]
    ax.legend(
        handles=patches, loc="lower right", fontsize=9,
        framealpha=0.9, title="Class", title_fontsize=10,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_component_grid(
    paths: dict,
    out_path: Path,
    source_label: str = "",
):
    """Four-panel grid showing all priority components side by side.

    paths: dict with keys 'heat', 'veg_deficit', 'built_up', 'equity',
           each pointing to a raster file.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    configs = [
        ("heat",        axes[0, 0], "Heat (LST °C)",        "hot",    None, None),
        ("veg_deficit", axes[0, 1], "NDVI",                 "RdYlGn", -0.2, 0.8),
        ("built_up",    axes[1, 0], "Built-up (land cover)", None,    None, None),
        ("equity",      axes[1, 1], "Equity (HVI 0–100)",   "viridis", 0,  100),
    ]

    for key, ax, title, cmap, vmin, vmax in configs:
        if key not in paths:
            ax.axis("off")
            ax.set_title(f"{title}\n(missing)", fontsize=12)
            continue

        data, extent = _read_raster(paths[key])
        if key == "built_up":
            # Categorical
            masked = np.ma.masked_where((data == 255) | np.isnan(data.astype(float)), data)
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], LANDCOVER_CMAP_3.N)
            ax.imshow(masked, cmap=LANDCOVER_CMAP_3, norm=norm, extent=extent, interpolation="nearest")
        else:
            masked = np.ma.masked_invalid(data)
            ax.imshow(masked, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    if source_label:
        fig.suptitle(source_label, fontsize=15, fontweight="bold", y=0.995)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)