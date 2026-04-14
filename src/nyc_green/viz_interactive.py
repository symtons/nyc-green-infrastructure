"""Folium interactive priority map — v1-style styling with sharp v2 raster overlays.

Generates a single self-contained HTML file with:
  - CartoDB Positron, OpenStreetMap, and CartoDB dark_matter basemaps
  - Three toggleable priority-tier overlays (Critical, High, Moderate) as
    sharp RGBA image overlays — one pixel per pixel, no blurring
  - A MarkerCluster of Critical-zone pins with rich popups (no dollar
    amounts; scientific impact numbers only)
  - Borough and MODZCTA-neighborhood lookup for each Critical zone pin
  - Title, stats, and legend overlays
  - Fullscreen button

Only the Model priority source is shown on this map. The WorldCover source
lives on the dashboard's Robustness page.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
from folium import plugins
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from shapely.geometry import Point


# ==========================================================================
# Priority category metadata (matches priority.py)
# ==========================================================================

PRIORITY_CODE_CRITICAL = 4
PRIORITY_CODE_HIGH     = 3
PRIORITY_CODE_MODERATE = 2
PRIORITY_CODE_LOW      = 1
PRIORITY_CODE_NONE     = 0

# Heat-intuitive palette: bright red / orange / yellow. Material Design
# 600-800 shades, chosen to stay distinct on both light and dark basemaps.
COLOR_CRITICAL = "#d32f2f"
COLOR_HIGH     = "#f57c00"
COLOR_MODERATE = "#fdd835"
COLOR_LOW      = "#fff59d"

PRIORITY_COLORS_HEX = {
    PRIORITY_CODE_CRITICAL: COLOR_CRITICAL,
    PRIORITY_CODE_HIGH:     COLOR_HIGH,
    PRIORITY_CODE_MODERATE: COLOR_MODERATE,
    PRIORITY_CODE_LOW:      COLOR_LOW,
}
PRIORITY_LABELS = {
    PRIORITY_CODE_CRITICAL: "Critical",
    PRIORITY_CODE_HIGH:     "High",
    PRIORITY_CODE_MODERATE: "Moderate",
    PRIORITY_CODE_LOW:      "Low",
}


# ==========================================================================
# Raster reprojection (UTM -> EPSG:4326 for Leaflet)
# ==========================================================================

def _reproject_to_4326(raster_path: Path):
    """Reproject a raster to EPSG:4326 and return (array, bounds, transform).

    bounds is (south, north, west, east) in degrees.
    transform is the rasterio Affine for converting pixel (row, col) to (lon, lat).
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

    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return dst, (bottom, top, left, right), transform


def _pixel_to_latlon(row: int, col: int, transform) -> tuple[float, float]:
    """Convert pixel coordinates to (lat, lon) using the rasterio transform."""
    lon, lat = rasterio.transform.xy(transform, row, col, offset="center")
    return float(lat), float(lon)


# ==========================================================================
# RGBA image overlay builder (one image per priority tier)
# ==========================================================================

def _build_tier_rgba(
    zones_4326: np.ndarray,
    priority_code: int,
    hex_color: str,
    alpha: int = 200,
) -> np.ndarray:
    """Build an (H, W, 4) uint8 RGBA image for a single priority tier.

    Pixels matching `priority_code` get the given hex_color at `alpha`.
    All other pixels are fully transparent. This preserves sharp NYC
    silhouette detail (unlike the point-based HeatMap approach which
    blurred everything into a single mass).
    """
    h, w = zones_4326.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    mask = zones_4326 == priority_code
    rgba[mask, 0] = r
    rgba[mask, 1] = g
    rgba[mask, 2] = b
    rgba[mask, 3] = alpha

    return rgba


# ==========================================================================
# Loading supplementary rasters for popup context
# ==========================================================================

def _load_context_raster(path: Path):
    """Load a raster into a (array, rasterio_dataset) tuple for coordinate lookup."""
    src = rasterio.open(path)
    data = src.read(1)
    return data, src


def _sample_context_at_latlon(
    lat: float,
    lon: float,
    data: np.ndarray,
    src,
) -> Optional[float]:
    """Sample a raster value at the given lat/lon coordinate.

    The raster is in UTM; we use rasterio's built-in coordinate transform
    to find the correct row/col. Returns None if out of bounds or nodata.
    """
    try:
        from rasterio.warp import transform as warp_transform
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        row, col = src.index(x, y)
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            val = data[row, col]
            if np.issubdtype(data.dtype, np.floating):
                if np.isnan(val):
                    return None
            else:
                if val == src.nodata:
                    return None
            return float(val)
    except Exception:
        return None
    return None


# ==========================================================================
# Borough / neighborhood lookup
# ==========================================================================

class _LocationLookup:
    """Point-in-polygon lookup for NYC boroughs and MODZCTAs.

    Built once per map build so we're not reopening/reparsing polygons on
    every single pin.
    """

    def __init__(
        self,
        borough_geojson: Optional[Path] = None,
        modzcta_geojson: Optional[Path] = None,
    ):
        self.boroughs = None
        self.modzctas = None

        if borough_geojson and borough_geojson.exists():
            try:
                gdf = gpd.read_file(borough_geojson)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                else:
                    gdf = gdf.to_crs("EPSG:4326")
                self.boroughs = gdf
            except Exception:
                self.boroughs = None

        if modzcta_geojson and modzcta_geojson.exists():
            try:
                gdf = gpd.read_file(modzcta_geojson)
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                else:
                    gdf = gdf.to_crs("EPSG:4326")
                self.modzctas = gdf
            except Exception:
                self.modzctas = None

    def lookup(self, lat: float, lon: float) -> dict:
        """Return a dict with 'borough' and 'neighborhood' (either may be None)."""
        pt = Point(lon, lat)
        out = {"borough": None, "neighborhood": None}

        if self.boroughs is not None:
            try:
                hit = self.boroughs[self.boroughs.geometry.contains(pt)]
                if len(hit) > 0:
                    for col in ("boroname", "borough", "BoroName", "BORONAME"):
                        if col in hit.columns:
                            out["borough"] = str(hit.iloc[0][col])
                            break
            except Exception:
                pass

        if self.modzctas is not None:
            try:
                hit = self.modzctas[self.modzctas.geometry.contains(pt)]
                if len(hit) > 0:
                    for col in ("label", "NEIGHBORHOOD_NAME", "neighborhood", "modzcta"):
                        if col in hit.columns:
                            out["neighborhood"] = str(hit.iloc[0][col])
                            break
            except Exception:
                pass

        return out


# ==========================================================================
# Critical-zone marker sampling
# ==========================================================================

def _sample_critical_pin_positions(
    zones_4326: np.ndarray,
    transform,
    pixels_per_pin: int,
) -> list:
    """Select representative Critical pixels for marker placement.

    Rather than every Nth pixel (which clusters in row-major order), we
    sample evenly across the Critical mask to get geographic spread.
    """
    mask = zones_4326 == PRIORITY_CODE_CRITICAL
    rows_idx, cols_idx = np.where(mask)
    n = len(rows_idx)
    if n == 0:
        return []

    n_pins = max(1, n // pixels_per_pin)
    step = n / n_pins
    idx = np.round(np.arange(n_pins) * step).astype(int)
    idx = np.clip(idx, 0, n - 1)

    positions = []
    for i in idx:
        r, c = int(rows_idx[i]), int(cols_idx[i])
        lat, lon = _pixel_to_latlon(r, c, transform)
        positions.append((lat, lon))
    return positions


def _build_popup_html(
    lat: float,
    lon: float,
    priority_score: Optional[float],
    lst_celsius: Optional[float],
    ndvi: Optional[float],
    hvi_scaled: Optional[float],
    location: dict,
) -> str:
    """Build the HTML for a Critical-zone popup.

    All numbers shown are scientific/factual, not financial. No dollar
    amounts, no cost estimates, no specific intervention quantities.
    """
    loc_bits = []
    if location.get("borough"):
        loc_bits.append(location["borough"])
    if location.get("neighborhood") and location["neighborhood"] != location.get("borough"):
        loc_bits.append(location["neighborhood"])
    location_line = " — ".join(loc_bits) if loc_bits else "NYC"

    score_str = f"{priority_score:.1f} / 100" if priority_score is not None else "unavailable"
    lst_str = f"{lst_celsius:.1f} &deg;C" if lst_celsius is not None else "unavailable"
    ndvi_str = f"{ndvi:.2f}" if ndvi is not None else "unavailable"
    hvi_str = f"{hvi_scaled:.0f} / 100" if hvi_scaled is not None else "unavailable"

    html = f"""
    <div style="font-family: 'Inter', -apple-system, system-ui, sans-serif;
                width: 300px; padding: 4px 8px; color: #222;">

      <div style="border-bottom: 2px solid {COLOR_CRITICAL};
                  padding-bottom: 6px; margin-bottom: 10px;">
        <div style="font-size: 13px; font-weight: 600; color: {COLOR_CRITICAL};
                    letter-spacing: 0.04em; text-transform: uppercase;">
          Critical Priority Zone
        </div>
        <div style="font-size: 14px; font-weight: 500; margin-top: 4px;">
          {location_line}
        </div>
        <div style="font-size: 11px; color: #666; margin-top: 2px;">
          {lat:.4f}, {lon:.4f}
        </div>
      </div>

      <div style="margin-bottom: 10px;">
        <div style="font-size: 11px; color: #666; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 4px;">
          Current conditions
        </div>
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
          <tr><td style="padding: 2px 0; color: #555;">Priority score</td>
              <td style="text-align: right; font-weight: 500;">{score_str}</td></tr>
          <tr><td style="padding: 2px 0; color: #555;">Land surface temp</td>
              <td style="text-align: right; font-weight: 500;">{lst_str}</td></tr>
          <tr><td style="padding: 2px 0; color: #555;">NDVI</td>
              <td style="text-align: right; font-weight: 500;">{ndvi_str}</td></tr>
          <tr><td style="padding: 2px 0; color: #555;">Heat vulnerability</td>
              <td style="text-align: right; font-weight: 500;">{hvi_str}</td></tr>
        </table>
      </div>

      <div style="margin-bottom: 10px; padding: 8px;
                  background: #fdf3f3; border-left: 3px solid {COLOR_CRITICAL};
                  border-radius: 3px;">
        <div style="font-size: 11px; color: {COLOR_CRITICAL}; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 4px; font-weight: 600;">
          Expected impact if greened
        </div>
        <div style="font-size: 12px; color: #444; line-height: 1.5;">
          Peak-summer surface temperature reduction: 2 to 4 &deg;C<br/>
          Carbon sequestration (dense canopy rate): 10 tCO<sub>2</sub>/ha/year<br/>
          Air-quality benefit from particulate and ozone uptake<br/>
          Shade and cooling co-benefits for heat-vulnerable residents
        </div>
      </div>

      <div style="font-size: 11px; color: #666; font-style: italic;
                  border-top: 1px solid #eee; padding-top: 6px;">
        Candidate for site-level assessment. Specific intervention type
        (canopy, green roof, pocket park) pending feasibility review.
      </div>
    </div>
    """
    return html


# ==========================================================================
# Main build function
# ==========================================================================

def build_priority_map(
    zones_paths: dict,
    out_path: Path,
    center: tuple = (40.73, -73.95),
    zoom_start: int = 11,
    lst_raster_path: Optional[Path] = None,
    landsat_raster_path: Optional[Path] = None,
    equity_raster_path: Optional[Path] = None,
    score_raster_path: Optional[Path] = None,
    borough_geojson: Optional[Path] = None,
    modzcta_geojson: Optional[Path] = None,
    pixels_per_critical_pin: int = 112,
) -> Path:
    """Build the full interactive priority map.

    zones_paths: dict — v2 passes {'WorldCover': ..., 'Model': ...} but
                 this function only uses the 'Model' entry for the visible
                 map. WorldCover is kept in the signature for backwards
                 compatibility with scripts/run_visualize.py.

    pixels_per_critical_pin: controls marker density. 112 pixels ≈ 10 ha
                             of Critical zone per pin.
    """
    # --- Pick the Model run as the visible source ---
    if "Model" in zones_paths:
        model_zones_path = zones_paths["Model"]
    else:
        model_zones_path = next(iter(zones_paths.values()))

    zones_4326, (south, north, west, east), ll_transform = _reproject_to_4326(
        model_zones_path
    )

    # --- Context rasters for popup content ---
    lst_data, lst_src = (None, None)
    if lst_raster_path and Path(lst_raster_path).exists():
        lst_data, lst_src = _load_context_raster(lst_raster_path)

    landsat_src = None
    red_band = nir_band = None
    if landsat_raster_path and Path(landsat_raster_path).exists():
        landsat_src = rasterio.open(landsat_raster_path)
        red_band = landsat_src.read(3).astype(np.float32)
        nir_band = landsat_src.read(4).astype(np.float32)

    equity_data, equity_src = (None, None)
    if equity_raster_path and Path(equity_raster_path).exists():
        equity_data, equity_src = _load_context_raster(equity_raster_path)

    score_data, score_src = (None, None)
    if score_raster_path and Path(score_raster_path).exists():
        score_data, score_src = _load_context_raster(score_raster_path)

    location_lookup = _LocationLookup(
        borough_geojson=borough_geojson,
        modzcta_geojson=modzcta_geojson,
    )

    # --- Base map ---
    m = folium.Map(
        location=list(center),
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark map").add_to(m)

    # --- Priority tier image overlays (one per tier, toggleable) ---
    bounds_4326 = [[south, west], [north, east]]

    if (zones_4326 == PRIORITY_CODE_CRITICAL).any():
        rgba_critical = _build_tier_rgba(
            zones_4326, PRIORITY_CODE_CRITICAL, COLOR_CRITICAL, alpha=230,
        )
        folium.raster_layers.ImageOverlay(
            image=rgba_critical,
            bounds=bounds_4326,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=3,
            name="Critical priority",
            show=True,
        ).add_to(m)

    if (zones_4326 == PRIORITY_CODE_HIGH).any():
        rgba_high = _build_tier_rgba(
            zones_4326, PRIORITY_CODE_HIGH, COLOR_HIGH, alpha=200,
        )
        folium.raster_layers.ImageOverlay(
            image=rgba_high,
            bounds=bounds_4326,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=2,
            name="High priority",
            show=True,
        ).add_to(m)

    if (zones_4326 == PRIORITY_CODE_MODERATE).any():
        rgba_moderate = _build_tier_rgba(
            zones_4326, PRIORITY_CODE_MODERATE, COLOR_MODERATE, alpha=170,
        )
        folium.raster_layers.ImageOverlay(
            image=rgba_moderate,
            bounds=bounds_4326,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=1,
            name="Moderate priority",
            show=False,
        ).add_to(m)

    # --- Critical-zone marker cluster with popups ---
    positions = _sample_critical_pin_positions(
        zones_4326, ll_transform, pixels_per_pin=pixels_per_critical_pin,
    )

    if positions:
        cluster = plugins.MarkerCluster(
            name=f"Critical zone details ({len(positions)} pins)",
            overlay=True,
            control=True,
            show=False,
        )

        for lat, lon in positions:
            score = _sample_context_at_latlon(lat, lon, score_data, score_src) if score_src else None
            lst_c = _sample_context_at_latlon(lat, lon, lst_data, lst_src) if lst_src else None
            hvi_s = _sample_context_at_latlon(lat, lon, equity_data, equity_src) if equity_src else None

            ndvi_v = None
            if landsat_src is not None and red_band is not None:
                try:
                    from rasterio.warp import transform as warp_transform
                    xs, ys = warp_transform("EPSG:4326", landsat_src.crs, [lon], [lat])
                    row, col = landsat_src.index(xs[0], ys[0])
                    if 0 <= row < red_band.shape[0] and 0 <= col < red_band.shape[1]:
                        r_val = red_band[row, col]
                        n_val = nir_band[row, col]
                        if np.isfinite(r_val) and np.isfinite(n_val) and (n_val + r_val) > 0:
                            ndvi_v = float((n_val - r_val) / (n_val + r_val + 1e-8))
                except Exception:
                    ndvi_v = None

            loc = location_lookup.lookup(lat, lon)

            popup_html = _build_popup_html(
                lat=lat, lon=lon,
                priority_score=score,
                lst_celsius=lst_c,
                ndvi=ndvi_v,
                hvi_scaled=hvi_s,
                location=loc,
            )

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=340),
                icon=folium.Icon(
                    color="darkred",
                    icon="exclamation-triangle",
                    prefix="fa",
                ),
            ).add_to(cluster)

        cluster.add_to(m)

    # Clean up rasterio handles
    if lst_src: lst_src.close()
    if landsat_src: landsat_src.close()
    if equity_src: equity_src.close()
    if score_src: score_src.close()

    # --- Layer control ---
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # --- Fullscreen button ---
    plugins.Fullscreen(
        position="topright",
        title="Fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(m)

    # --- Title box (top-left) ---
    n_crit = int((zones_4326 == PRIORITY_CODE_CRITICAL).sum())
    n_high = int((zones_4326 == PRIORITY_CODE_HIGH).sum())
    n_mod  = int((zones_4326 == PRIORITY_CODE_MODERATE).sum())

    ha_crit = round(n_crit * 900.0 / 10_000.0)
    ha_high = round(n_high * 900.0 / 10_000.0)
    ha_mod  = round(n_mod  * 900.0 / 10_000.0)

    title_html = """
    <div style="position: fixed; top: 12px; left: 60px; z-index: 9999;
                background: rgba(14, 17, 23, 0.95);
                border: 1px solid #2a2f3d; border-radius: 6px;
                padding: 12px 16px;
                font-family: 'Inter', -apple-system, system-ui, sans-serif;
                color: #e8eaed; box-shadow: 0 2px 12px rgba(0,0,0,0.4);">
        <div style="font-size: 15px; font-weight: 600; letter-spacing: -0.01em;">
            NYC Green Infrastructure Priority
        </div>
        <div style="font-size: 11px; color: #9aa0a6; margin-top: 3px;
                    letter-spacing: 0.03em;">
            Five boroughs &middot; Equity-weighted priority scoring
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # --- Stats box (top-left, below title) ---
    stats_html = f"""
    <div style="position: fixed; top: 86px; left: 60px; z-index: 9999;
                background: rgba(14, 17, 23, 0.95);
                border: 1px solid #2a2f3d; border-radius: 6px;
                padding: 10px 14px; width: 210px;
                font-family: 'Inter', sans-serif; color: #e8eaed;
                box-shadow: 0 2px 12px rgba(0,0,0,0.4);">
        <div style="font-size: 10px; color: #9aa0a6; text-transform: uppercase;
                    letter-spacing: 0.06em; font-weight: 500; margin-bottom: 6px;">
            Priority zone area
        </div>
        <div style="display: flex; justify-content: space-between;
                    font-size: 12px; padding: 2px 0;">
            <span style="display: flex; align-items: center;">
                <span style="display: inline-block; width: 10px; height: 10px;
                             background: {COLOR_CRITICAL}; border-radius: 2px; margin-right: 7px;"></span>
                Critical
            </span>
            <span style="font-weight: 500;">{ha_crit:,} ha</span>
        </div>
        <div style="display: flex; justify-content: space-between;
                    font-size: 12px; padding: 2px 0;">
            <span style="display: flex; align-items: center;">
                <span style="display: inline-block; width: 10px; height: 10px;
                             background: {COLOR_HIGH}; border-radius: 2px; margin-right: 7px;"></span>
                High
            </span>
            <span style="font-weight: 500;">{ha_high:,} ha</span>
        </div>
        <div style="display: flex; justify-content: space-between;
                    font-size: 12px; padding: 2px 0;">
            <span style="display: flex; align-items: center;">
                <span style="display: inline-block; width: 10px; height: 10px;
                             background: {COLOR_MODERATE}; border-radius: 2px; margin-right: 7px;"></span>
                Moderate
            </span>
            <span style="font-weight: 500;">{ha_mod:,} ha</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    # --- Legend (bottom-left) ---
    legend_html = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999;
                background: rgba(14, 17, 23, 0.95);
                border: 1px solid #2a2f3d; border-radius: 6px;
                padding: 12px 14px; width: 220px;
                font-family: 'Inter', sans-serif; color: #e8eaed;
                box-shadow: 0 2px 12px rgba(0,0,0,0.4);">
        <div style="font-size: 11px; color: #9aa0a6; text-transform: uppercase;
                    letter-spacing: 0.06em; font-weight: 500; margin-bottom: 8px;">
            Priority tier
        </div>
        <div style="font-size: 12px; line-height: 1.7;">
            <div><span style="display:inline-block;width:14px;height:10px;
                 background:{COLOR_CRITICAL};vertical-align:middle;margin-right:8px;
                 border-radius:2px;"></span>
                 Critical &middot; top 5%</div>
            <div><span style="display:inline-block;width:14px;height:10px;
                 background:{COLOR_HIGH};vertical-align:middle;margin-right:8px;
                 border-radius:2px;"></span>
                 High &middot; next 10%</div>
            <div><span style="display:inline-block;width:14px;height:10px;
                 background:{COLOR_MODERATE};vertical-align:middle;margin-right:8px;
                 border-radius:2px;"></span>
                 Moderate &middot; next 20%</div>
        </div>
        <div style="margin-top: 10px; padding-top: 8px;
                    border-top: 1px solid #2a2f3d;
                    font-size: 10px; color: #9aa0a6; line-height: 1.5;">
            Score weights: heat 25%, vegetation deficit 30%,
            built-up 20%, equity (HVI) 25%.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path