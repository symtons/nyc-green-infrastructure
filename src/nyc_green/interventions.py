"""Step 10 — Intervention recommendations based on priority zones.

Translates Critical and High priority areas into concrete intervention portfolios:
street trees, green roofs, and pocket parks. Computes total cost, expected
carbon sequestration, and cost-effectiveness metrics.

Coefficients are deliberately conservative and sourced from common NYC urban
forestry rates. In a real consultancy deliverable these would be replaced
with a client-specific unit cost assumption file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import rasterio


# ==========================================================================
# Intervention coefficients
# ==========================================================================
# All coefficients are per-unit, conservative, and meant to be swappable.

STREET_TREE_COST_USD       = 900.0     # planted + 2 years of care
STREET_TREE_DENSITY_PER_HA = 20        # mature-tree spacing
STREET_TREE_CARBON_T_PER_YR = 0.60     # per tree, per year

GREEN_ROOF_COST_USD_PER_M2  = 100.0
GREEN_ROOF_CARBON_T_PER_M2  = 0.015
GREEN_ROOF_FRACTION_OF_HA   = 0.10     # 10% of the priority area goes to roofs

POCKET_PARK_AREA_HA         = 0.65
POCKET_PARK_COST_USD        = 80000.0
POCKET_PARK_CARBON_T_PER_YR = 6.5


@dataclass
class InterventionRow:
    intervention: str
    quantity: str
    cost_usd: float
    carbon_t_per_year: float
    cost_per_t_carbon: float


def recommend_for_area(
    critical_ha: float,
    high_ha: float,
) -> pd.DataFrame:
    """Build an intervention portfolio for Critical + High priority areas.

    Strategy:
      - Critical area gets the full treatment: street trees at full density,
        green roofs on 10% of the area, and pocket parks to fill gaps.
      - High area gets street trees only at half density (lower-urgency,
        maintain existing green fabric rather than create new).
    """
    rows: List[InterventionRow] = []

    # --- Critical: full intervention ---
    if critical_ha > 0:
        crit_trees = int(round(critical_ha * STREET_TREE_DENSITY_PER_HA))
        crit_tree_cost = crit_trees * STREET_TREE_COST_USD
        crit_tree_carbon = crit_trees * STREET_TREE_CARBON_T_PER_YR
        rows.append(InterventionRow(
            intervention="Street trees (Critical)",
            quantity=f"{crit_trees:,} trees",
            cost_usd=crit_tree_cost,
            carbon_t_per_year=crit_tree_carbon,
            cost_per_t_carbon=crit_tree_cost / max(crit_tree_carbon, 1e-9),
        ))

        roof_ha = critical_ha * GREEN_ROOF_FRACTION_OF_HA
        roof_m2 = roof_ha * 10_000.0
        roof_cost = roof_m2 * GREEN_ROOF_COST_USD_PER_M2
        roof_carbon = roof_m2 * GREEN_ROOF_CARBON_T_PER_M2
        rows.append(InterventionRow(
            intervention="Green roofs (Critical)",
            quantity=f"{roof_ha:.1f} ha ({roof_m2:,.0f} m²)",
            cost_usd=roof_cost,
            carbon_t_per_year=roof_carbon,
            cost_per_t_carbon=roof_cost / max(roof_carbon, 1e-9),
        ))

        n_parks = int(round(critical_ha / POCKET_PARK_AREA_HA))
        park_cost = n_parks * POCKET_PARK_COST_USD
        park_carbon = n_parks * POCKET_PARK_CARBON_T_PER_YR
        rows.append(InterventionRow(
            intervention="Pocket parks (Critical)",
            quantity=f"{n_parks} parks",
            cost_usd=park_cost,
            carbon_t_per_year=park_carbon,
            cost_per_t_carbon=park_cost / max(park_carbon, 1e-9),
        ))

    # --- High: half-density street trees only ---
    if high_ha > 0:
        high_trees = int(round(high_ha * STREET_TREE_DENSITY_PER_HA * 0.5))
        high_tree_cost = high_trees * STREET_TREE_COST_USD
        high_tree_carbon = high_trees * STREET_TREE_CARBON_T_PER_YR
        rows.append(InterventionRow(
            intervention="Street trees (High)",
            quantity=f"{high_trees:,} trees",
            cost_usd=high_tree_cost,
            carbon_t_per_year=high_tree_carbon,
            cost_per_t_carbon=high_tree_cost / max(high_tree_carbon, 1e-9),
        ))

    df = pd.DataFrame([r.__dict__ for r in rows])

    # Total row
    if not df.empty:
        total = pd.DataFrame([{
            "intervention": "TOTAL",
            "quantity": "—",
            "cost_usd": df["cost_usd"].sum(),
            "carbon_t_per_year": df["carbon_t_per_year"].sum(),
            "cost_per_t_carbon": df["cost_usd"].sum() / max(df["carbon_t_per_year"].sum(), 1e-9),
        }])
        df = pd.concat([df, total], ignore_index=True)

    return df


def compute_area_ha_from_zones(zones_path, priority_level: int, pixel_area_m2: float = 900.0) -> float:
    """Count pixels of a given priority level and return the area in hectares."""
    with rasterio.open(zones_path) as src:
        data = src.read(1)
    n = int((data == priority_level).sum())
    return n * pixel_area_m2 / 10_000.0