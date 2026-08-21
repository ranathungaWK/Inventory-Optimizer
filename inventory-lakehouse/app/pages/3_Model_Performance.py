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
        load_store_metrics,
        load_family_metrics,
        load_demand_tier_metrics,
        load_dow_metrics,
        load_promo_metrics,
        load_holiday_metrics,
        load_perishability_metrics,
        load_hyperparameter_results,
        load_model_comparison
    )
    from app.utils.plots import (
        plot_slice_bar,
        plot_optuna_convergence,
        plot_boosting_iterations_tradeoff,
        plot_training_loss_progression
    )
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
        load_store_metrics,
        load_family_metrics,
        load_demand_tier_metrics,
        load_dow_metrics,
        load_promo_metrics,
        load_holiday_metrics,
        load_perishability_metrics,
        load_hyperparameter_results,
        load_model_comparison
    )
    from utils.plots import (
        plot_slice_bar,
        plot_optuna_convergence,
        plot_boosting_iterations_tradeoff,
        plot_training_loss_progression
    )

render_global_header("Model Performance & Slice Benchmarks")

# 1. Load Contracts & Datasets
mc = load_model_contract()
diag = load_diagnostic_report()
scoring = load_scoring_contract()

store_df = load_store_metrics()
family_df = load_family_metrics()
tier_df = load_demand_tier_metrics()
dow_df = load_dow_metrics()
promo_df = load_promo_metrics()
holiday_df = load_holiday_metrics()
perish_df = load_perishability_metrics()
hyper_df = load_hyperparameter_results()
comp_df = load_model_comparison()

# 2. Render Governance Status
render_governance_banner(mc, diag, scoring)

# 3. Macro Benchmarking Summary Cards
render_section_header("Global Benchmark Summary (Holdout Test Split)", "target", "blue", "Holdout evaluation against naive and heuristic moving average baselines")

m1, m2, m3, m4 = st.columns(4)

with m1:
    render_metric_card(
        label="LightGBM Final WMAPE",
        value="44.92%",
        delta="Primary Model",
        delta_type="up",
        subtitle="19.57M test rows evaluated"
    )

with m2:
    render_metric_card(
        label="MA-7 Moving Average Baseline",
        value="48.59%",
        delta="+3.67% Error vs LightGBM",
        delta_type="down",
        subtitle="Gate Requirement: Met"
    )

with m3:
    render_metric_card(
        label="LightGBM Final MAE",
        value="3.90 units",
        delta="-0.53 units vs Baseline",
        delta_type="up",
        subtitle="Baseline MAE: 4.43 units"
    )

with m4:
    render_metric_card(
        label="Mean Global Bias",
        value="+0.14 units",
        delta="Near-Zero Error Mean",
        delta_type="neutral",
        subtitle="Symmetric residual center"
    )

# 4. Multi-Slice Breakdown Tabs
render_section_header("Multi-Dimensional Performance & Training Evolution", "layers", "indigo", "Detailed error slices across operations, merchandising, seasonality, and Optuna tuning")

tab_train, tab_store, tab_fam, tab_tier, tab_dow, tab_promo, tab_hol, tab_perish = st.tabs([
    "Training & Boosting Evolution",
    "Stores (54)",
    "Product Families (31)",
    "Demand Tiers",
    "Day of Week",
    "Promotions",
    "Holidays",
    "Perishability"
])

# Tab 0: Training & Boosting Evolution
with tab_train:
    render_section_header("Hyperparameter Optimization & Boosting Round Convergence", "trending_up", "emerald", "Loss minimization and learning rate progression across 10 Optuna trials and 496 boosting rounds", heading_level=4)
    
    if not hyper_df.empty:
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            fig_conv = plot_optuna_convergence(hyper_df)
            st.plotly_chart(fig_conv, use_container_width=True, config={"displayModeBar": False})
        with c_tr2:
            fig_loss = plot_training_loss_progression(best_iteration=496)
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})
            
        render_section_header("Boosting Rounds (Tree Count) vs Validation Error Tradeoff", "tree", "purple", heading_level=4)
        c_tr3, c_tr4 = st.columns([7, 5])
        with c_tr3:
            fig_trade = plot_boosting_iterations_tradeoff(hyper_df)
            st.plotly_chart(fig_trade, use_container_width=True, config={"displayModeBar": False})
        with c_tr4:
            st.markdown("##### Optuna 10-Trial Optimization Ledger")
            disp_hyper = hyper_df[["trial", "val_wmape", "val_mae", "best_iteration", "learning_rate", "num_leaves"]].copy()
            disp_hyper["val_wmape"] = (disp_hyper["val_wmape"] * 100).round(2).astype(str) + "%"
            disp_hyper["val_mae"] = disp_hyper["val_mae"].round(3)
            disp_hyper["learning_rate"] = disp_hyper["learning_rate"].round(4)
            st.dataframe(disp_hyper.sort_values("val_wmape"), use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.6rem; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.22); border-radius:8px; padding:0.65rem 0.9rem; margin-top:0.6rem; font-size:0.82rem; color:var(--text);">
                {render_icon_badge("award", "blue", size=14, padding="4px")}
                <div><strong>Winning Configuration (Trial 7):</strong> 496 trees, learning rate <code>0.0269</code>, 127 leaves, achieving lowest validation WMAPE (<code>43.75%</code>).</div>
            </div>
            """, unsafe_allow_html=True)

# Tab 1: By Store
with tab_store:
    render_section_header("Store-Level Forecast Accuracy & Error Contribution", "bar_chart", "blue", heading_level=4)
    if not store_df.empty:
        c_left, c_right = st.columns([6, 6])
        with c_left:
            fig_store_wmape = plot_slice_bar(
                df=store_df.sort_values("wmape", ascending=False),
                x_col="store_id",
                y_col="wmape",
                title="Top Stores with Highest Error (WMAPE)",
                x_label="Store ID",
                y_label="WMAPE",
                top_n=15
            )
            st.plotly_chart(fig_store_wmape, use_container_width=True, config={"displayModeBar": False})
        with c_right:
            st.markdown("##### Store Error Metrics Table")
            st.dataframe(
                store_df[["store_id", "wmape", "mae", "bias", "error_contribution_pct", "sales_share_pct", "reliability_flag"]],
                use_container_width=True,
                hide_index=True
            )

# Tab 2: By Family
with tab_fam:
    render_section_header("Merchandise Category Performance", "package", "cyan", heading_level=4)
    if not family_df.empty:
        c_left, c_right = st.columns([6, 6])
        with c_left:
            fig_fam_wmape = plot_slice_bar(
                df=family_df.sort_values("wmape", ascending=False),
                x_col="family",
                y_col="wmape",
                title="Top Categories with Highest WMAPE",
                x_label="Family",
                y_label="WMAPE",
                top_n=15
            )
            st.plotly_chart(fig_fam_wmape, use_container_width=True, config={"displayModeBar": False})
        with c_right:
            st.markdown("##### Category Error Metrics Table")
            st.dataframe(
                family_df[["family", "wmape", "mae", "bias", "error_contribution_pct", "sales_share_pct", "reliability_flag"]],
                use_container_width=True,
                hide_index=True
            )

# Tab 3: By Demand Tier
with tab_tier:
    render_section_header("Demand Volume Tier Error Decomposition", "sliders", "purple", heading_level=4)
    if not tier_df.empty:
        st.dataframe(tier_df, use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.22); border-radius:8px; padding:0.65rem 0.9rem; margin-top:0.6rem; font-size:0.82rem; color:var(--text);">
            {render_icon_badge("search", "blue", size=14, padding="4px")}
            <div><strong>Insight:</strong> Low-volume items show higher percentage error (WMAPE 56.1%) but small absolute error (MAE 1.56). High-volume items have lower WMAPE (43.9%) with larger scale (MAE 8.44).</div>
        </div>
        """, unsafe_allow_html=True)

# Tab 4: Day of Week
with tab_dow:
    render_section_header("Day of Week Seasonality Performance", "bar_chart", "emerald", heading_level=4)
    if not dow_df.empty:
        st.dataframe(dow_df, use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.22); border-radius:8px; padding:0.65rem 0.9rem; margin-top:0.6rem; font-size:0.82rem; color:var(--text);">
            {render_icon_badge("search", "blue", size=14, padding="4px")}
            <div><strong>Insight:</strong> Weekend shopping spikes (Saturday/Sunday) maintain robust accuracy (WMAPE ~44-46%) due to strong day-of-week lag features (<code>lag_7</code>, <code>lag_14</code>, <code>lag_28</code>).</div>
        </div>
        """, unsafe_allow_html=True)

# Tab 5: Promotions
with tab_promo:
    render_section_header("Promotional Uplift Interaction Performance", "target", "amber", heading_level=4)
    if not promo_df.empty:
        st.dataframe(promo_df, use_container_width=True, hide_index=True)

# Tab 6: Holidays
with tab_hol:
    render_section_header("National & Regional Holiday Performance", "sparkles", "indigo", heading_level=4)
    if not holiday_df.empty:
        st.dataframe(holiday_df, use_container_width=True, hide_index=True)

# Tab 7: Perishability
with tab_perish:
    render_section_header("Perishable vs Non-Perishable Inventory Performance", "scale", "rose", heading_level=4)
    if not perish_df.empty:
        st.dataframe(perish_df, use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.22); border-radius:8px; padding:0.65rem 0.9rem; margin-top:0.6rem; font-size:0.82rem; color:var(--text);">
            {render_icon_badge("scale", "rose", size=14, padding="4px")}
            <div><strong>Insight:</strong> Perishable merchandise contributes 23.5% of total absolute error, making empirical safety stock buffers critical to prevent spoilage.</div>
        </div>
        """, unsafe_allow_html=True)
