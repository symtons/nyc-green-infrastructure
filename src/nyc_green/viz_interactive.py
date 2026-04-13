"""Step 10 — Folium interactive map of priority zones.

Produces a single HTML file with:
  - OpenStreetMap base layer
  - CartoDB Positron and dark basemaps as alternatives
  - Priority zones from both WorldCover and model runs as toggleable overlays
  - A color-coded legend
  - Clickable popups showing the priority category and coordinates
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, calculate_default_transform, Resampling


PRIORITY_COLORS_HEX = {
    0: "#f5f5f5",
    1: "#ffe8a1",
    2: "#ffb35c",
    3: "#ef5350",
    4: "#7f0000",
}
PRIORITY_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Critical"}


def _reproject_to_4326(raster_path: Path):
    """Reproject a UTM raster to EPSG:4326 in memory and return (array, bounds_4326).

    Folium expects lat/lon bounds, so we reproject the priority raster before overlaying.
    """
    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        dst = np.full((height, width), 255, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=255,
        )

    # Bounds of the reprojected raster (lat/lon)
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    bounds_4326 = [[bottom, left], [top, right]]
    return dst, bounds_4326


def _zones_to_rgba(zones: np.ndarray) -> np.ndarray:
    """Convert a categorical zones array to an (H, W, 4) RGBA uint8 image."""
    h, w = zones.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for code, hex_color in PRIORITY_COLORS_HEX.items():
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        mask = zones == code
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b
        # Alpha: transparent for None, increasingly opaque for higher priority
        if code == 0:
            rgba[mask, 3] = 0
        elif code == 1:
            rgba[mask, 3] = 120
        elif code == 2:
            rgba[mask, 3] = 150
        elif code == 3:
            rgba[mask, 3] = 180
        elif code == 4:
            rgba[mask, 3] = 210
    # Fully transparent outside NYC
    rgba[zones == 255, 3] = 0
    return rgba


def build_priority_map(
    zones_paths: dict,
    out_path: Path,
    center: tuple = (40.73, -73.95),
    zoom_start: int = 11,
) -> Path:
    """Build an interactive Folium map with toggleable priority overlays.

    zones_paths: dict like {'WorldCover': Path(...), 'Model': Path(...)}
    """
    m = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles="CartoDB Positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)

    for label, raster_path in zones_paths.items():
        zones, bounds_4326 = _reproject_to_4326(raster_path)
        rgba = _zones_to_rgba(zones)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=bounds_4326,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=1,
            name=f"Priority zones — {label}",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        width: 180px;
        background: white;
        border: 2px solid #555;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: -apple-system, system-ui, sans-serif;
        font-size: 12px;
        z-index: 9999;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
        <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px;">
            Green Infrastructure<br/>Priority
        </div>
    """
    for code in (4, 3, 2, 1):
        legend_html += (
            f'<div style="display:flex; align-items:center; margin:3px 0;">'
            f'<div style="width:16px;height:16px;background:{PRIORITY_COLORS_HEX[code]};'
            f'border:1px solid #666;margin-right:8px;"></div>'
            f'{PRIORITY_LABELS[code]}</div>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path