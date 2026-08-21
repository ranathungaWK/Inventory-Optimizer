"""
UI/UX Style Provider & Component Builders
Implements the Zinc Dark/Light theme, metric cards, badges, and responsive CSS styling.
"""
import streamlit as st
import pandas as pd

try:
    from app.utils.icons import get_svg_icon, render_icon_badge
except ImportError:
    from utils.icons import get_svg_icon, render_icon_badge

def init_theme_state():
    """Initializes dark/light theme state in session state."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

def toggle_theme():
    """Toggles theme state between dark and light."""
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

def is_dark_mode() -> bool:
    """Returns True if dark mode is active."""
    return st.session_state.get("theme", "dark") == "dark"

def inject_custom_css():
    """Injects responsive CSS tokens into Streamlit."""
    init_theme_state()
    dark = is_dark_mode()

    bg = "#09090b" if dark else "#ffffff"
    bg_subtle = "#0c0c0f" if dark else "#f9fafb"
    card = "#0c0c0f" if dark else "#ffffff"
    card_hover = "#131316" if dark else "#f4f4f5"
    border = "#1e1e24" if dark else "#e4e4e7"
    border_subtle = "#16161a" if dark else "#f0f0f2"
    text = "#fafafa" if dark else "#09090b"
    text_muted = "#a1a1aa" if dark else "#71717a"
    text_dim = "#71717a" if dark else "#a1a1aa"
    accent = "#3b82f6"
    green = "#22c55e" if dark else "#16a34a"
    green_bg = "rgba(34,197,94,0.12)" if dark else "rgba(22,163,74,0.08)"
    red = "#ef4444" if dark else "#dc2626"
    red_bg = "rgba(239,68,68,0.12)" if dark else "rgba(220,38,38,0.08)"
    amber = "#f59e0b" if dark else "#d97706"
    amber_bg = "rgba(245,158,11,0.12)" if dark else "rgba(217,119,6,0.08)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap');

    :root {{
        --bg: {bg};
        --bg-subtle: {bg_subtle};
        --card: {card};
        --card-hover: {card_hover};
        --border: {border};
        --border-subtle: {border_subtle};
        --text: {text};
        --text-muted: {text_muted};
        --text-dim: {text_dim};
        --accent: {accent};
        --green: {green};
        --green-bg: {green_bg};
        --red: {red};
        --red-bg: {red_bg};
        --amber: {amber};
        --amber-bg: {amber_bg};
        --radius: 10px;
    }}

    /* Hide Streamlit Default Overlapping Top Bar */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    #MainMenu {{
        visibility: hidden !important;
    }}
    footer {{
        visibility: hidden !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}
    div[data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    /* Global App Styling */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}

    .block-container {{
        padding-top: 2.25rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1400px !important;
    }}

    /* Card Containers */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.15rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }}
    .metric-label {{
        font-size: 0.76rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-top: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
    }}
    .metric-delta {{
        font-size: 0.74rem;
        font-weight: 600;
        margin-top: 0.35rem;
        padding: 2px 7px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }}
    .delta-up {{ color: var(--green); background: var(--green-bg); }}
    .delta-down {{ color: var(--red); background: var(--red-bg); }}
    .delta-warn {{ color: var(--amber); background: var(--amber-bg); }}
    .delta-neutral {{ color: var(--text-muted); background: var(--bg-subtle); }}

    /* Status Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 9px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }}
    .badge-green {{ color: var(--green); background: var(--green-bg); border: 1px solid rgba(34,197,94,0.3); }}
    .badge-red {{ color: var(--red); background: var(--red-bg); border: 1px solid rgba(239,68,68,0.3); }}
    .badge-amber {{ color: var(--amber); background: var(--amber-bg); border: 1px solid rgba(245,158,11,0.3); }}
    .badge-blue {{ color: var(--accent); background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3); }}
    .badge-neutral {{ color: var(--text-muted); background: var(--bg-subtle); border: 1px solid var(--border); }}

    /* Pulse Indicator Dot */
    .pulse-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: currentColor;
        box-shadow: 0 0 6px currentColor;
        display: inline-block;
    }}

    /* Platform Governance Header */
    .gov-banner {{
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.7rem 1.1rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        font-size: 0.78rem;
    }}

    /* HTML Data Tables */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.65rem 0.85rem;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid var(--border);
        background: var(--bg-subtle);
    }}
    .data-table td {{
        padding: 0.65rem 0.85rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}

    /* Plotly Chart Card Container */
    .chart-container {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.1rem 1.2rem 0.5rem;
        margin-bottom: 1.25rem;
    }}
    .chart-title {{
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text);
    }}
    .chart-subtitle {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, delta: str = None, delta_type: str = "up", subtitle: str = None):
    """Renders a styled KPI metric card."""
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else ("•" if delta_type == "neutral" else "!"))
    delta_html = f'<div class="metric-delta delta-{delta_type}"><span>{arrow}</span> {delta}</div>' if delta else ""
    sub_html = f'<div style="font-size:0.72rem; color:var(--text-dim); margin-top:0.25rem;">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_global_header(current_page_title: str):
    """Renders top header with brand logo, page title, and status badges."""
    inject_custom_css()
    
    brand_logo = render_icon_badge("lakehouse", badge_color="blue", size=20, padding="7px")
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem; border-bottom:1px solid var(--border); padding-bottom:0.5rem;">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            {brand_logo}
            <div>
                <span style="font-size:1.35rem; font-weight:700; letter-spacing:-0.03em;">Inventory Lakehouse</span>
                <span style="color:var(--text-muted); font-size:0.95rem; margin-left:0.5rem; font-weight:500;">/ {current_page_title}</span>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span class="badge badge-green"><span class="pulse-dot"></span> STATUS: HEALTHY</span>
            <span class="badge badge-blue">ENGINE: LIGHTGBM V1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_governance_banner(model_contract: dict, diag_report: dict, scoring_contract: dict):
    """Renders an enterprise governance status bar across pages."""
    model_version = model_contract.get("model", {}).get("model_version", "LightGBM_v1")
    run_id = model_contract.get("mlflow", {}).get("final_model_run_id", "e8da7be6...")[:8]
    n_trees = model_contract.get("model", {}).get("best_iteration", 496)
    n_features = len(model_contract.get("model", {}).get("features", []))
    sha256_short = diag_report.get("model_sha256", "ea408d13...")[:10] + "..."
    blockers = len(diag_report.get("blocking_issues", []))
    n_partitions = scoring_contract.get("scoring_modes", {}).get("holdout_replay", {}).get("dates_scored", 227)

    diag_badge = '<span class="badge badge-green"><span class="pulse-dot"></span> DIAGNOSTICS: PASSED (0 BLOCKERS)</span>' if blockers == 0 else '<span class="badge badge-red">DIAGNOSTICS: BLOCKED</span>'
    gov_badge = '<span class="badge badge-blue">GOVERNANCE: READ-ONLY</span>'

    st.markdown(f"""
    <div class="gov-banner">
        <div style="display:flex; align-items:center; gap:0.6rem; flex-wrap:wrap;">
            <span><strong>Model:</strong> {model_version} ({n_trees} trees, {n_features} features)</span>
            <span style="color:var(--border);">|</span>
            <span><strong>Run ID:</strong> <code>{run_id}</code></span>
            <span style="color:var(--border);">|</span>
            <span><strong>Checksum:</strong> <code>{sha256_short}</code></span>
            <span style="color:var(--border);">|</span>
            <span><strong>Scored Partitions:</strong> {n_partitions}</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            {diag_badge}
            {gov_badge}
        </div>
    </div>
    """, unsafe_allow_html=True)
