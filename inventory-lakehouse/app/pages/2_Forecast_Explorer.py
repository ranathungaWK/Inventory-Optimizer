import sys
from pathlib import Path

# Ensure both project root and app directory are on sys.path
pages_dir = Path(__file__).resolve().parent
app_dir = pages_dir.parent
project_root = app_dir.parent
for p in [str(project_root), str(app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

try:
    from app.utils.style import (
        render_global_header, 
        render_governance_banner, 
        render_metric_card
    )
    from app.utils.icons import (
        render_section_header,
        render_icon_badge
    )
    from app.utils.data_loader import (
        load_model_contract, 
        load_diagnostic_report, 
        load_scoring_contract,
        get_distinct_store_metadata,
        get_distinct_families,
        get_distinct_skus,
        load_store_sku_forecast_series
    )
    from app.utils.plots import plot_forecast_series, plot_newsvendor_cost_curve
except ImportError:
    from utils.style import (
        render_global_header, 
        render_governance_banner, 
        render_metric_card
    )
    from utils.icons import (
        render_section_header,
        render_icon_badge
    )
    from utils.data_loader import (
        load_model_contract, 
        load_diagnostic_report, 
        load_scoring_contract,
        get_distinct_store_metadata,
        get_distinct_families,
        get_distinct_skus,
        load_store_sku_forecast_series
    )
    from utils.plots import plot_forecast_series, plot_newsvendor_cost_curve

render_global_header("Forecast Explorer & Newsvendor Optimizer")

# 1. Load Governance Contracts
mc = load_model_contract()
diag = load_diagnostic_report()
scoring = load_scoring_contract()

render_governance_banner(mc, diag, scoring)

# 2. Interactive Selection Controls
render_section_header("Store & SKU Demand Forecaster", "search", "blue", "Filter by retail location, merchandise department, and item SKU")

store_meta_df = get_distinct_store_metadata()

c_store, c_fam, c_sku = st.columns([4, 4, 3])

with c_store:
    store_options = store_meta_df.apply(
        lambda r: f"Store {r['store_id']} ({r['city']}, Type {r['store_type']}, Cluster {r['cluster']})", 
        axis=1
    ).tolist()
    store_id_map = {opt: row["store_id"] for opt, (_, row) in zip(store_options, store_meta_df.iterrows())}
    selected_store_label = st.selectbox("Store Location:", options=store_options, index=0)
    selected_store_id = store_id_map[selected_store_label]

with c_fam:
    available_families = get_distinct_families(selected_store_id)
    default_fam_idx = available_families.index("GROCERY I") if "GROCERY I" in available_families else 0
    selected_family = st.selectbox("Product Family (Category):", options=available_families, index=default_fam_idx)

with c_sku:
    available_skus = get_distinct_skus(selected_store_id, selected_family)
    selected_sku_id = st.selectbox("Product SKU ID:", options=available_skus, index=0 if available_skus else None)

if not selected_sku_id:
    st.warning("No active SKUs found for the selected Store and Category.")
    st.stop()

# 3. Load Scored Forecast Series via DuckDB
with st.spinner(f"Loading forecasts for Store {selected_store_id}, SKU {selected_sku_id}..."):
    series_df = load_store_sku_forecast_series(selected_store_id, selected_sku_id)

if series_df.empty:
    st.info("No forecast records available for this item in the holdout partition.")
    st.stop()

# 4. Interactive Newsvendor Decision Simulator (User-Defined Assumptions)
st.markdown("---")
render_section_header("Interactive Newsvendor Inventory Decision Simulator", "scale", "emerald", "Configure unit economics to compute optimal replenishment order sizing (Q*) and safety stock buffers")

# User-Defined Financial Parameters
sim_c1, sim_c2, sim_c3 = st.columns(3)

with sim_c1:
    unit_price = st.number_input("Unit Retail Price ($P):", min_value=1.0, max_value=500.0, value=10.0, step=0.5)
with sim_c2:
    unit_cost = st.number_input("Unit Purchase Cost ($C):", min_value=0.5, max_value=unit_price, value=6.0, step=0.5)
with sim_c3:
    salvage_val = st.number_input("Salvage / Markdown Value ($V):", min_value=0.0, max_value=unit_cost, value=2.0, step=0.5)

# Calculate Cost Parameters
Cu = unit_price - unit_cost      # Understock cost (lost margin)
Co = unit_cost - salvage_val     # Overstock cost (markdown / spoilage loss)
critical_fractile = Cu / (Cu + Co)

# Latest forecast values for this item
latest_row = series_df.iloc[-1]
y_hat = float(latest_row["predicted_unit_sales"])
q80_upper = float(latest_row["operational_upper_80"])
q80_lower = float(latest_row["operational_lower_80"])
tier = str(latest_row["demand_tier"])

# Sizing calculation:
if abs(critical_fractile - 0.80) < 0.02:
    q_star = q80_upper
    calc_mode_note = "Exact Empirical 80% Quantile from NB06"
else:
    sigma_approx = max((q80_upper - y_hat) / 1.28, 0.5)
    z_score = norm.ppf(critical_fractile)
    q_star = max(0.0, y_hat + z_score * sigma_approx)
    calc_mode_note = f"Approximated Quantile for SL*={critical_fractile:.1%}"

safety_stock = max(0.0, q_star - y_hat)

# Display Calculated Replenishment Cards
render_section_header("Replenishment Decision Outputs (Latest Origin: 2017-08-15)", "target", "purple", "Optimal stock ordering parameters driven by critical fractile economics", heading_level=4)
rc1, rc2, rc3, rc4 = st.columns(4)

with rc1:
    render_metric_card(
        label="Critical Fractile (Target SL*)",
        value=f"{critical_fractile:.1%}",
        delta=f"Cu=${Cu:.2f} / Co=${Co:.2f}",
        delta_type="neutral",
        subtitle="Optimal Service Level"
    )

with rc2:
    render_metric_card(
        label="Mean Forecast (ŷ)",
        value=f"{y_hat:.1f} units",
        delta=f"Tier: {tier}",
        delta_type="neutral",
        subtitle=f"80% Bounds: [{q80_lower:.1f}, {q80_upper:.1f}]"
    )

with rc3:
    render_metric_card(
        label="Recommended Order (Q*)",
        value=f"{q_star:.1f} units",
        delta=f"{calc_mode_note}",
        delta_type="up",
        subtitle="Newsvendor Optimal Quantity"
    )

with rc4:
    render_metric_card(
        label="Safety Stock Buffer (SS)",
        value=f"{safety_stock:.1f} units",
        delta=f"Buffer: +{(safety_stock/max(y_hat,0.1))*100:.0f}% over ŷ",
        delta_type="neutral",
        subtitle="Stockout Protection Margin"
    )

# 5. Visual Time-Series & Expected Cost Curves
col_ts, col_cost = st.columns([7, 5])

with col_ts:
    fig_ts = plot_forecast_series(
        df=series_df,
        sku_id=selected_sku_id,
        store_id=selected_store_id,
        recommended_q=q_star
    )
    st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})

with col_cost:
    fig_cost = plot_newsvendor_cost_curve(
        y_hat=y_hat,
        Cu=Cu,
        Co=Co,
        q80_upper=q80_upper
    )
    st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})

# 6. Granular Forecast Table & Export
render_section_header("Granular Scored Forecast Table", "table", "indigo", "Point predictions, empirical uncertainty bounds, and expected error estimates")

display_cols = [
    "target_date", "forecast_date", "predicted_unit_sales", "actual_unit_sales",
    "demand_tier", "operational_lower_80", "operational_upper_80", "expected_error_mae"
]
available_disp = [c for c in display_cols if c in series_df.columns]
tbl_df = series_df[available_disp].sort_values("target_date", ascending=False)

st.dataframe(tbl_df, use_container_width=True, hide_index=True)

# Export button
csv_data = tbl_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Export Scored Forecasts as CSV",
    data=csv_data,
    file_name=f"forecast_store_{selected_store_id}_sku_{selected_sku_id}.csv",
    mime="text/csv"
)
