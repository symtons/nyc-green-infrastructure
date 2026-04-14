"""NYC Green Infrastructure Priority Analysis — Streamlit dashboard.

Five pages that visualize the outputs of the Step 1-10 pipeline without
recomputing anything. Run with:

    streamlit run app.py

All data is read from outputs/, models/, and docs/ as produced by the
pipeline scripts. If a page shows "no data available", run the corresponding
pipeline step first.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Make the src package importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nyc_green.dashboard_helpers import (
    get_headline_metrics,
    get_model_summary,
    get_priority_component_paths,
    get_zone_summary_df,
    get_cutoffs,
    get_score_stats,
    load_methodology_markdown,
    load_priority_summary,
    figure_path,
    interactive_map_path,
)


# ==========================================================================
# Page configuration and global styles
# ==========================================================================

st.set_page_config(
    page_title="NYC Green Infrastructure Priority",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

h1 {
    font-size: 2.1rem !important;
    margin-bottom: 0.3rem !important;
}

.metric-tile {
    background: #161a23;
    border: 1px solid #2a2f3d;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}

.metric-tile .label {
    font-size: 0.78rem;
    color: #9aa0a6;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
    margin-bottom: 0.4rem;
}

.metric-tile .value {
    font-size: 2rem;
    font-weight: 600;
    color: #e8eaed;
    line-height: 1.1;
}

.metric-tile .caption {
    font-size: 0.75rem;
    color: #6e7480;
    margin-top: 0.3rem;
}

.section-divider {
    border-top: 1px solid #2a2f3d;
    margin: 2rem 0 1.4rem 0;
}

.subtle {
    color: #9aa0a6;
    font-size: 0.9rem;
}

/* Remove Streamlit default footer branding */
footer { visibility: hidden; }

/* Tighter sidebar */
section[data-testid="stSidebar"] {
    background: #0a0d14;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# Reusable UI components
# ==========================================================================

def metric_tile(label: str, value: str, caption: str = ""):
    html = f"""
    <div class="metric-tile">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {f'<div class="caption">{caption}</div>' if caption else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)


# ==========================================================================
# Sidebar navigation
# ==========================================================================

st.sidebar.markdown("### NYC Green Infrastructure")
st.sidebar.markdown(
    '<div class="subtle">Priority analysis with equity weighting</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("")

PAGES = [
    "Overview",
    "Model",
    "Priority Scoring",
    "Robustness",
    "Methodology",
]

page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

st.sidebar.markdown("")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="subtle" style="font-size:0.78rem;">'
    "Data: Landsat 9 (2024), ESA WorldCover v200, NYC DOHMH Heat Vulnerability Index, "
    "NYC DCP borough boundaries."
    "</div>",
    unsafe_allow_html=True,
)


# ==========================================================================
# PAGE 1 — OVERVIEW
# ==========================================================================

if page == "Overview":
    st.markdown("# NYC Green Infrastructure Priority Analysis")
    st.markdown(
        '<div class="subtle">'
        "Identifying where across the five boroughs green infrastructure investment "
        "would do the most good, weighted by physical heat stress and community vulnerability."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    metrics = get_headline_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        crit = metrics.get("critical_ha")
        metric_tile(
            "Critical priority area",
            f"{crit:,.0f} ha" if crit is not None else "—",
            "Top 2% of NYC by composite need",
        )
    with col2:
        high = metrics.get("high_ha")
        metric_tile(
            "High priority area",
            f"{high:,.0f} ha" if high is not None else "—",
            "Percentile rank 85–95",
        )
    with col3:
        iou = metrics.get("test_mean_iou")
        metric_tile(
            "Test set mean IoU",
            f"{iou:.3f}" if iou is not None else "—",
            "U-Net, held-out 30 tiles",
        )
    with col4:
        agr = metrics.get("agreement_pct")
        metric_tile(
            "Robustness to land cover",
            f"{agr:.1f}%" if agr is not None else "—",
            "WorldCover vs Model category agreement",
        )

    divider()

    section_header(
        "Interactive priority map",
        "Pan and zoom to explore. Use the layer control (top right) to toggle "
        "between the two land cover sources and basemap styles.",
    )

    map_file = interactive_map_path()
    if map_file.exists():
        html = map_file.read_text(encoding="utf-8")
        components.html(html, height=640, scrolling=False)
    else:
        st.warning("Interactive map not found. Run `python scripts/run_visualize.py` first.")

    divider()

    section_header("What changed from v1")
    improvements = pd.DataFrame([
        {
            "Dimension": "Study area",
            "v1": "Manhattan + Brooklyn (2 boroughs)",
            "v2": "All 5 boroughs",
        },
        {
            "Dimension": "Classes",
            "v1": "4 (vegetation / water / built-up / bare)",
            "v2": "3 (bare dropped with evidence, <1% coverage)",
        },
        {
            "Dimension": "Encoder weights",
            "v1": "Trained from scratch",
            "v2": "ImageNet pretrained, 5-channel adapted",
        },
        {
            "Dimension": "Class imbalance handling",
            "v1": "Focal loss only",
            "v2": "Focal loss + weighted sampler on water",
        },
        {
            "Dimension": "Primary metric",
            "v1": "Overall accuracy",
            "v2": "Per-class IoU",
        },
        {
            "Dimension": "Scoring inputs",
            "v1": "Heat, vegetation, built-up",
            "v2": "Heat, vegetation, built-up + NYC Heat Vulnerability Index",
        },
        {
            "Dimension": "Priority thresholds",
            "v1": "Fixed absolute cutoffs",
            "v2": "Percentile-based, calibrated to distribution",
        },
        {
            "Dimension": "Land cover source",
            "v1": "Single source",
            "v2": "Dual run (WorldCover + U-Net), robustness reported",
        },
    ])
    st.dataframe(improvements, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE 2 — MODEL
# ==========================================================================

elif page == "Model":
    st.markdown("# Model")
    st.markdown(
        '<div class="subtle">'
        "U-Net land cover segmentation trained on NYC Landsat tiles with ImageNet "
        "transfer learning and class-aware sampling."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    model_data = get_model_summary()

    if not model_data:
        st.warning(
            "No training history found. Run `python scripts/run_train.py` first."
        )
    else:
        # --- Architecture summary ---
        section_header("Architecture")
        arch_md = """
- **Backbone:** U-Net encoder-decoder with ResNet18 encoder
- **Pretraining:** ImageNet weights on the first three input channels (BGR),
  with NIR and NDVI channels Kaiming-initialized and scaled by 3/5 so total
  activation magnitudes are preserved
- **Input:** 5 channels × 256 × 256 pixels
- **Output:** 3-class segmentation (vegetation, water, built-up)
- **Loss:** Focal loss with per-class alpha `[1.0, 1.5, 0.8]`, gamma 2.0
- **Sampling:** Weighted random sampler, water-fraction boost of 5
- **Optimizer:** Adam, learning rate 1e-4, ReduceLROnPlateau
"""
        st.markdown(arch_md)

        divider()

        # --- Test metrics tiles ---
        section_header("Held-out test set results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            acc = model_data.get("test_accuracy")
            metric_tile("Accuracy", f"{acc:.3f}" if acc else "—")
        with c2:
            iou = model_data.get("test_mean_iou")
            metric_tile("Mean IoU", f"{iou:.3f}" if iou else "—")
        with c3:
            t = model_data.get("total_time_sec")
            metric_tile(
                "Training time",
                f"{t:.0f} s" if t else "—",
                "RTX 4080 Laptop GPU",
            )
        with c4:
            metric_tile("Training tiles", "139", "Split 70/15/15 from 199 total")

        divider()

        # --- Per-class IoU bar chart ---
        section_header("Per-class IoU")
        iou_per_class = model_data.get("test_iou_per_class") or [0, 0, 0]
        class_names = ["Vegetation", "Water", "Built-up"]
        iou_df = pd.DataFrame({"Class": class_names, "IoU": iou_per_class})
        fig = px.bar(
            iou_df,
            x="Class",
            y="IoU",
            text=[f"{v:.3f}" for v in iou_per_class],
            color="Class",
            color_discrete_map={
                "Vegetation": "#66bb6a",
                "Water": "#42a5f5",
                "Built-up": "#bdbdbd",
            },
        )
        fig.update_traces(textposition="outside", textfont_size=13)
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(family="Inter", color="#e8eaed"),
            showlegend=False,
            yaxis=dict(range=[0, 1], gridcolor="#2a2f3d", title=""),
            xaxis=dict(title=""),
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        divider()

        # --- Training curves ---
        section_header("Training curves")
        history = model_data.get("history") or []
        if history:
            hdf = pd.DataFrame(history)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hdf["epoch"], y=hdf["train_loss"],
                name="Train loss",
                line=dict(color="#64b5f6", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=hdf["epoch"], y=hdf["val_loss"],
                name="Val loss",
                line=dict(color="#ef5350", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=hdf["epoch"], y=hdf["val_mean_iou"],
                name="Val mean IoU",
                line=dict(color="#66bb6a", width=2, dash="dot"),
                yaxis="y2",
            ))
            fig2.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(family="Inter", color="#e8eaed"),
                xaxis=dict(title="Epoch", gridcolor="#2a2f3d"),
                yaxis=dict(title="Loss", gridcolor="#2a2f3d"),
                yaxis2=dict(
                    title="Mean IoU",
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    gridcolor="#2a2f3d",
                ),
                legend=dict(orientation="h", y=-0.2),
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
            )
            st.plotly_chart(fig2, use_container_width=True)

        divider()

        # --- Confusion matrix ---
        section_header("Confusion matrix", "Rows: ground truth. Columns: model prediction.")
        cm = model_data.get("confusion_matrix")
        if cm:
            cm_array = np.array(cm)
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm_array,
                x=class_names,
                y=class_names,
                text=[[f"{v:,}" for v in row] for row in cm_array],
                texttemplate="%{text}",
                textfont=dict(family="Inter", size=13, color="#e8eaed"),
                colorscale=[
                    [0.0, "#0e1117"],
                    [0.5, "#3d2a37"],
                    [1.0, "#ef5350"],
                ],
                showscale=False,
            ))
            cm_fig.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(family="Inter", color="#e8eaed"),
                xaxis=dict(title="Predicted", side="bottom"),
                yaxis=dict(title="True", autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        divider()

        section_header("Class imbalance decision")
        st.markdown(
            "The bare/open class was dropped after initial analysis showed NYC has "
            "under 1% bare land and zero training tiles with more than 5% bare coverage. "
            "A model cannot learn a class that does not appear in the data at any "
            "meaningful density. The former bare/open WorldCover codes (cropland, "
            "bare/sparse vegetation, snow, moss) were merged into built-up, which "
            "matches how they visually appear in dense NYC — gravel lots, construction "
            "sites, stadium fields. This reformulation is an evidence-based response "
            "to a class that would otherwise silently collapse to zero predictions."
        )


# ==========================================================================
# PAGE 3 — PRIORITY SCORING
# ==========================================================================

elif page == "Priority Scoring":
    st.markdown("# Priority Scoring")
    st.markdown(
        '<div class="subtle">'
        "Combining physical heat stress, vegetation deficit, built-up density, and "
        "community vulnerability into a single per-pixel priority score."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # --- Formula ---
    section_header("Scoring formula")
    st.latex(r"""
    \text{priority} =
    0.25 \cdot \text{heat}
    + 0.30 \cdot \text{vegetation deficit}
    + 0.20 \cdot \text{built-up}
    + 0.25 \cdot \text{equity}
    """)
    st.markdown(
        "Each component is rescaled to 0–100 before weighting. Categories are assigned "
        "by percentile — top 5% Critical, next 10% High, next 20% Moderate, next 25% Low — "
        "so cutoffs adapt to the score distribution rather than assuming fixed thresholds."
    )

    divider()

    # --- Weight donut ---
    section_header("Component weights")
    weights_df = pd.DataFrame({
        "Component": ["Vegetation deficit", "Heat", "Equity (HVI)", "Built-up"],
        "Weight": [0.30, 0.25, 0.25, 0.20],
    })
    donut = px.pie(
        weights_df,
        values="Weight",
        names="Component",
        hole=0.55,
        color="Component",
        color_discrete_map={
            "Vegetation deficit": "#66bb6a",
            "Heat": "#ef5350",
            "Equity (HVI)": "#ab47bc",
            "Built-up": "#bdbdbd",
        },
    )
    donut.update_traces(
        textinfo="label+percent",
        textfont=dict(family="Inter", size=13),
    )
    donut.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(family="Inter", color="#e8eaed"),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    st.plotly_chart(donut, use_container_width=True)

    divider()

    # --- Component context maps (PNG figures) ---
    section_header("Component inputs")
    comp_paths = get_priority_component_paths()
    c1, c2 = st.columns(2)
    with c1:
        if comp_paths["heat"].exists():
            st.image(str(comp_paths["heat"]), caption="Land surface temperature (summer 2024)")
        if comp_paths["landcover_m"].exists():
            st.image(str(comp_paths["landcover_m"]), caption="Land cover — U-Net predictions")
    with c2:
        if comp_paths["equity"].exists():
            st.image(str(comp_paths["equity"]), caption="NYC Heat Vulnerability Index")
        if comp_paths["landcover_w"].exists():
            st.image(str(comp_paths["landcover_w"]), caption="Land cover — ESA WorldCover")

    divider()

    # --- Zone summary table ---
    section_header("Priority zone distribution (model run)")
    zone_df = get_zone_summary_df("model")
    if zone_df is not None:
        display_df = zone_df.copy()
        display_df["Percent"] = display_df["Percent"].apply(lambda x: f"{x:.2f}%")
        display_df["Area (ha)"] = display_df["Area (ha)"].apply(lambda x: f"{x:,.0f}")
        display_df["Pixels"] = display_df["Pixels"].apply(lambda x: f"{x:,}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    cutoffs = get_cutoffs("model")
    if cutoffs:
        st.markdown(
            f'<div class="subtle">'
            f"Score cutoffs (percentile-calibrated): "
            f"Critical ≥ {cutoffs['critical']:.1f}, "
            f"High ≥ {cutoffs['high']:.1f}, "
            f"Moderate ≥ {cutoffs['moderate']:.1f}, "
            f"Low ≥ {cutoffs['low']:.1f}."
            f"</div>",
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE 4 — ROBUSTNESS
# ==========================================================================

elif page == "Robustness":
    st.markdown("# Robustness to land cover source")
    st.markdown(
        '<div class="subtle">'
        "The priority scoring pipeline was run twice — once with ESA WorldCover as the "
        "land cover input, once with the trained U-Net predictions. Both runs used "
        "identical heat, NDVI, and equity inputs."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    summary = load_priority_summary()
    if summary is None:
        st.warning("No priority summary found. Run `python scripts/run_priority.py` first.")
    else:
        comparison = summary.get("comparison", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            pct = comparison.get("exact_agreement_pct", 0)
            metric_tile(
                "Exact category agreement",
                f"{pct:.2f}%",
                "Pixel-level match across all five priority categories",
            )
        with c2:
            pct2 = comparison.get("within_one_pct", 0)
            metric_tile(
                "Within one category",
                f"{pct2:.2f}%",
                "Pixels differing by at most one category",
            )
        with c3:
            tot = comparison.get("total_compared", 0)
            metric_tile(
                "Pixels compared",
                f"{tot:,}",
                "All valid NYC land pixels in both runs",
            )

        divider()

        # --- Side-by-side priority maps ---
        section_header("Priority maps side by side")
        c1, c2 = st.columns(2)
        wc_fig = figure_path("priority_zones_worldcover.png")
        m_fig = figure_path("priority_zones_model.png")
        with c1:
            if wc_fig.exists():
                st.image(str(wc_fig), caption="Priority zones — ESA WorldCover input")
        with c2:
            if m_fig.exists():
                st.image(str(m_fig), caption="Priority zones — U-Net model input")

        divider()

        # --- Zone area comparison ---
        section_header("Zone area comparison")
        wc_df = get_zone_summary_df("worldcover")
        m_df = get_zone_summary_df("model")
        if wc_df is not None and m_df is not None:
            merged = pd.merge(
                wc_df[["Priority", "Area (ha)"]].rename(columns={"Area (ha)": "WorldCover (ha)"}),
                m_df[["Priority", "Area (ha)"]].rename(columns={"Area (ha)": "Model (ha)"}),
                on="Priority",
                how="outer",
            )
            merged["Difference (ha)"] = (merged["Model (ha)"] - merged["WorldCover (ha)"]).round(1)
            merged["WorldCover (ha)"] = merged["WorldCover (ha)"].apply(lambda x: f"{x:,.1f}")
            merged["Model (ha)"] = merged["Model (ha)"].apply(lambda x: f"{x:,.1f}")
            st.dataframe(merged, use_container_width=True, hide_index=True)

        divider()

        section_header("What this means")
        st.markdown(
            "The two land cover sources produce near-identical priority category "
            "assignments. Differences at the individual component level (the model "
            "predicts slightly more built-up than WorldCover) are absorbed by the "
            "combined heat, vegetation, and equity signal, which dominates the priority "
            "score. For a decision-support context, this robustness check is more "
            "valuable than either source's absolute land cover accuracy: it indicates "
            "that a planner using these outputs does not need to commit to one land "
            "cover source to trust the conclusions."
        )


# ==========================================================================
# PAGE 5 — METHODOLOGY
# ==========================================================================

elif page == "Methodology":
    st.markdown("# Methodology")
    st.markdown(
        '<div class="subtle">'
        "Design rationale for each step of the pipeline."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    md = load_methodology_markdown()
    if md is None:
        st.warning("docs/methodology.md not found.")
    else:
        st.markdown(md)