"""Helper functions for the Streamlit dashboard.

Loads data from the existing pipeline outputs (JSON files, PNGs, rasters)
and formats it for display. The dashboard is a pure view layer — no
computation happens here, just reading and minor reshaping.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================================
# JSON loaders
# ==========================================================================

def load_training_history() -> Optional[dict]:
    """Load training_history.json from models/ if present."""
    path = PROJECT_ROOT / "models" / "training_history.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_priority_summary() -> Optional[dict]:
    """Load priority_summary.json from outputs/analysis/ if present."""
    path = PROJECT_ROOT / "outputs" / "analysis" / "priority_summary.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_model_vs_worldcover() -> Optional[dict]:
    """Load model_vs_worldcover.json from outputs/analysis/ if present."""
    path = PROJECT_ROOT / "outputs" / "analysis" / "model_vs_worldcover.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_methodology_markdown() -> Optional[str]:
    """Load docs/methodology.md if present."""
    path = PROJECT_ROOT / "docs" / "methodology.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


# ==========================================================================
# Paths for figures and maps (used by st.image / components.html)
# ==========================================================================

def figure_path(name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "figures" / name


def map_path(name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "maps" / name


def interactive_map_path() -> Path:
    return PROJECT_ROOT / "outputs" / "maps" / "priority_zones_interactive.html"


# ==========================================================================
# Headline metrics
# ==========================================================================

def get_headline_metrics() -> dict:
    """Compute the four numbers shown on the Overview page."""
    summary = load_priority_summary()
    history = load_training_history()

    out = {
        "critical_ha": None,
        "high_ha": None,
        "test_mean_iou": None,
        "agreement_pct": None,
    }

    if summary and "runs" in summary:
        model_run = summary["runs"].get("model", {})
        zones = model_run.get("zone_summary", {})
        if "Critical" in zones:
            out["critical_ha"] = zones["Critical"].get("area_ha")
        if "High" in zones:
            out["high_ha"] = zones["High"].get("area_ha")

    if summary and "comparison" in summary:
        out["agreement_pct"] = summary["comparison"].get("exact_agreement_pct")

    if history and "test_metrics" in history:
        out["test_mean_iou"] = history["test_metrics"].get("mean_iou")

    return out


def get_model_summary() -> dict:
    """Architecture + training metrics for the Model page."""
    history = load_training_history()
    if history is None:
        return {}
    test_m = history.get("test_metrics", {}) or {}
    return {
        "test_accuracy": test_m.get("accuracy"),
        "test_mean_iou": test_m.get("mean_iou"),
        "test_iou_per_class": test_m.get("iou_per_class"),
        "confusion_matrix": test_m.get("confusion_matrix"),
        "history": history.get("history", []),
        "total_time_sec": history.get("total_time_sec"),
    }


def get_priority_component_paths() -> dict:
    """Return paths for the four priority component figures on the Priority page."""
    return {
        "heat":        figure_path("context_lst.png"),
        "equity":      figure_path("context_equity.png"),
        "landcover_m": figure_path("landcover_model.png"),
        "landcover_w": figure_path("landcover_worldcover.png"),
    }


def get_zone_summary_df(run_label: str = "model") -> Optional[pd.DataFrame]:
    """Zone area summary for either 'model' or 'worldcover' run."""
    summary = load_priority_summary()
    if not summary:
        return None
    run = summary["runs"].get(run_label, {})
    zone_summary = run.get("zone_summary", {})
    if not zone_summary:
        return None
    rows = []
    for name, data in zone_summary.items():
        rows.append({
            "Priority": name,
            "Pixels": data.get("pixels", 0),
            "Percent": data.get("percent", 0.0),
            "Area (ha)": data.get("area_ha", 0.0),
        })
    df = pd.DataFrame(rows)
    # Order: Critical, High, Moderate, Low, None
    order = ["Critical", "High", "Moderate", "Low", "None / Excluded"]
    df["__sort"] = df["Priority"].apply(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("__sort").drop(columns="__sort").reset_index(drop=True)
    return df


def get_cutoffs(run_label: str = "model") -> Optional[dict]:
    """Percentile cutoffs used by the priority scoring."""
    summary = load_priority_summary()
    if not summary:
        return None
    run = summary["runs"].get(run_label, {})
    return run.get("cutoffs_used")


def get_score_stats(run_label: str = "model") -> Optional[dict]:
    """Descriptive stats for the continuous priority score."""
    summary = load_priority_summary()
    if not summary:
        return None
    run = summary["runs"].get(run_label, {})
    return run.get("score_stats")