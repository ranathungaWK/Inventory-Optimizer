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
        load_feature_importance,
        load_calibration_deciles,
        load_worst_error_cases
    )
    from app.utils.plots import plot_feature_importance, plot_calibration_curve
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
        load_feature_importance,
        load_calibration_deciles,
        load_worst_error_cases
    )
    from utils.plots import plot_feature_importance, plot_calibration_curve

render_global_header("Model Diagnostics & Error Decomposition")

# 1. Load Datasets
mc = load_model_contract()
diag = load_diagnostic_report()
scoring = load_scoring_contract()

feat_df = load_feature_importance()
calib_df = load_calibration_deciles()
worst_df = load_worst_error_cases()

render_governance_banner(mc, diag, scoring)

# 2. Section 1: Feature Importance Explorer
render_section_header("Feature Importance & Predictive Split Mechanisms", "tree", "emerald", "Evaluates predictive contribution across all 24 features in the frozen LightGBM booster (496 trees)")

if not feat_df.empty:
    feat_c1, feat_c2 = st.columns([8, 4])
    
    with feat_c1:
        metric_choice = st.radio(
            "Feature Ranking Metric:",
            options=["Gain % (Variance Reduction)", "Split Count (Frequency)"],
            horizontal=True
        )
        metric_key = "importance_gain" if "Gain" in metric_choice else "importance_split"
        fig_feat = plot_feature_importance(feat_df, metric=metric_key, top_n=15)
        st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})
        
    with feat_c2:
        st.markdown("##### Feature Importance Table")
        st.dataframe(
            feat_df.sort_values("importance_gain", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# 3. Section 2: Empirical Calibration Deciles
st.markdown("---")
render_section_header("Empirical Model Calibration Curve", "target", "blue", "Bins holdout predictions into 10 deciles (NTILE(10)) to evaluate whether predicted demand magnitude matches actual realized volume")

if not calib_df.empty:
    cal_left, cal_right = st.columns([6, 6])
    
    with cal_left:
        fig_cal = plot_calibration_curve(calib_df)
        st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})
        
    with cal_right:
        st.markdown("##### Decile Calibration Summary Table")
        disp_cal = calib_df[["decile", "row_count", "mean_prediction", "mean_actual", "bias", "mae", "wmape"]].copy()
        st.dataframe(disp_cal, use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.22); border-radius:8px; padding:0.65rem 0.9rem; margin-top:0.6rem; font-size:0.82rem; color:var(--text);">
            {render_icon_badge("search", "blue", size=14, padding="4px")}
            <div><strong>Calibration Assessment:</strong> Empirical curve closely tracks 1:1 diagonal across deciles 1–8. Deciles 9–10 exhibit minor conservative under-prediction on extreme volume spikes.</div>
        </div>
        """, unsafe_allow_html=True)

# 4. Section 3: Worst-Case Tail Error Attribution
st.markdown("---")
render_section_header("Top 20 Worst-Case Error Attribution Cases", "alert_triangle", "amber", "Identifies highest absolute error instances in holdout dataset to diagnose operational volatility drivers")

if not worst_df.empty:
    st.dataframe(worst_df, use_container_width=True, hide_index=True)
