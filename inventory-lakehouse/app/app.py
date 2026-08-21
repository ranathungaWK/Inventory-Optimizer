import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Inventory Demand Forecasting Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure both project root and app directory are on sys.path
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
for p in [str(project_root), str(app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.utils.style import (
        inject_custom_css, 
        render_global_header, 
        render_governance_banner, 
        render_metric_card
    )
    from app.utils.icons import (
        get_svg_icon,
        render_icon_badge,
        render_section_header
    )
    from app.utils.data_loader import (
        load_model_contract, 
        load_diagnostic_report, 
        load_scoring_contract,
        load_scoring_manifest,
        load_baseline_contract
    )
    from app.utils.validators import validate_pipeline_integrity
except ImportError:
    from utils.style import (
        inject_custom_css, 
        render_global_header, 
        render_governance_banner, 
        render_metric_card
    )
    from utils.icons import (
        get_svg_icon,
        render_icon_badge,
        render_section_header
    )
    from utils.data_loader import (
        load_model_contract, 
        load_diagnostic_report, 
        load_scoring_contract,
        load_scoring_manifest,
        load_baseline_contract
    )
    from utils.validators import validate_pipeline_integrity

# 1. Inject Theme & Layout Styling
inject_custom_css()

# 2. Load Core Pipeline Contracts
mc = load_model_contract()
bc = load_baseline_contract()
diag = load_diagnostic_report()
scoring = load_scoring_contract()
manifest = load_scoring_manifest()

# 3. Render Top Global Navigation Header
render_global_header("Command Center")

# 4. Render Governance Status Banner
render_governance_banner(mc, diag, scoring)

# 5. Application Overview & Architecture Introduction
st.markdown("""
<div style="margin-top: 1rem; margin-bottom: 1.5rem;">
    <h2 style="font-weight: 700; letter-spacing: -0.03em; margin-bottom: 0.25rem;">
        Enterprise Demand Forecasting & Inventory Optimizer
    </h2>
    <p style="color: var(--text-muted); font-size: 0.92rem; max-width: 900px;">
        An end-to-end Lakehouse intelligence platform delivering D &rarr; D+1 retail demand forecasts, 
        statistically calibrated uncertainty intervals, and interactive Newsvendor replenishment optimization 
        across 54 stores and 3,500+ SKUs.
    </p>
</div>
""", unsafe_allow_html=True)

# 6. High-Level Pipeline Key Metrics
c1, c2, c3, c4 = st.columns(4)

with c1:
    test_wmape = mc.get("model_evaluation", {}).get("test_wmape", 0.4492)
    base_wmape = bc.get("results", {}).get("test_wmape", 0.4859)
    rel_imp_wmape = ((base_wmape - test_wmape) / base_wmape) * 100
    render_metric_card(
        label="Official Test WMAPE",
        value=f"{test_wmape:.2%}",
        delta=f"{rel_imp_wmape:.2f}% vs MA-7 Baseline",
        delta_type="up",
        subtitle=f"Baseline WMAPE: {base_wmape:.2%}"
    )

with c2:
    test_mae = mc.get("model_evaluation", {}).get("test_mae", 3.8977)
    base_mae = bc.get("results", {}).get("test_mae", 4.4300)
    rel_imp_mae = ((base_mae - test_mae) / base_mae) * 100
    render_metric_card(
        label="Official Test MAE",
        value=f"{test_mae:.2f} units",
        delta=f"{rel_imp_mae:.2f}% Error Reduction",
        delta_type="up",
        subtitle=f"Baseline MAE: {base_mae:.2f} units"
    )

with c3:
    n_rows = scoring.get("scoring_modes", {}).get("holdout_replay", {}).get("total_rows", 19571966)
    n_dates = scoring.get("scoring_modes", {}).get("holdout_replay", {}).get("dates_scored", 227)
    render_metric_card(
        label="Holdout Forecast Volume",
        value=f"{n_rows / 1e6:.2f}M rows",
        delta=f"{n_dates} Partitions Scored",
        delta_type="neutral",
        subtitle="2017-01-01 to 2017-08-15"
    )

with c4:
    op_skus = scoring.get("scoring_modes", {}).get("operational_simulation", {}).get("active_store_skus", 84659)
    render_metric_card(
        label="Active Operational SKUs",
        value=f"{op_skus:,}",
        delta="Latest Origin: 2017-08-14",
        delta_type="neutral",
        subtitle="54 Stores / 31 Categories"
    )

# 7. Navigation Quick Links with Modern Vector Badges
render_section_header("Platform Modules", "compass", "indigo", "Access operational dashboards and specialized modeling tools")

badge_1 = render_icon_badge("bar_chart", "blue", size=18, padding="7px")
badge_2 = render_icon_badge("trending_up", "emerald", size=18, padding="7px")
badge_3 = render_icon_badge("target", "purple", size=18, padding="7px")
badge_4 = render_icon_badge("microscope", "amber", size=18, padding="7px")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            {badge_1}
            <span style="font-weight:600; font-size:0.95rem; color:var(--text);">1. Executive Overview</span>
        </div>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:0;">
            Macro-level model performance comparisons, business KPI cards, and latest operational demand volume distribution.
        </p>
    </div>
    <div class="metric-card">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            {badge_2}
            <span style="font-weight:600; font-size:0.95rem; color:var(--text);">2. Forecast Explorer & Newsvendor Simulator</span>
        </div>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:0;">
            Interactive store/SKU forecast inspection with <strong>empirical 80% uncertainty bounds</strong> and interactive 
            <strong>Newsvendor inventory replenishment order sizing (Q*)</strong> under user-defined cost assumptions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            {badge_3}
            <span style="font-weight:600; font-size:0.95rem; color:var(--text);">3. Model Performance & Benchmarks</span>
        </div>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:0;">
            Multi-slice statistical benchmarks across all 54 stores, 31 product families, volume tiers, day of week, promotions, and holidays.
        </p>
    </div>
    <div class="metric-card">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
            {badge_4}
            <span style="font-weight:600; font-size:0.95rem; color:var(--text);">4. Diagnostics & Error Decomposition</span>
        </div>
        <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:0;">
            Gain % vs Split feature importance, empirical decile calibration curves, and worst-case error attribution cases.
        </p>
    </div>
    """, unsafe_allow_html=True)

# 8. Integrity Validation Status
val = validate_pipeline_integrity()
if val["is_healthy"]:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.65rem; background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25); border-radius:8px; padding:0.75rem 1rem; margin-top:1.25rem;">
        {render_icon_badge("check_circle", "green", size=16, padding="5px")}
        <div style="font-size:0.85rem; color:var(--text);">
            <strong>Pipeline Verified:</strong> Model SHA-256 <code>{val['model_sha256'][:16]}...</code> matched, {val['total_partitions']} date partitions valid, zero technical blockers.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.65rem; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); border-radius:8px; padding:0.75rem 1rem; margin-top:1.25rem;">
        {render_icon_badge("alert_triangle", "amber", size=16, padding="5px")}
        <div style="font-size:0.85rem; color:var(--text);">
            <strong>Pipeline Warnings:</strong> {', '.join(val['issues'])}
        </div>
    </div>
    """, unsafe_allow_html=True)
