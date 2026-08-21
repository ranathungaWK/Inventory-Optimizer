"""
Modern Vector SVG Icon System & Clipart Badges
Provides theme-aware, resolution-independent SVG vector glyphs and duotone badge containers.
Replaces OS emojis with enterprise-grade micro-visuals.
"""

from typing import Optional
import streamlit as st

# Precision SVG Vector Paths (Lucide / Heroicon inspired, 24x24 viewBox)
SVG_PATHS = {
    # Platform Brand / Architecture
    "lakehouse": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
    </svg>''',
    
    "cube": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.12 6.4-6-3.87a2.53 2.53 0 0 0-2.24 0l-6 3.87A2.5 2.5 0 0 0 6 8.5v7a2.5 2.5 0 0 0 1.12 2.1l6 3.87a2.52 2.52 0 0 0 2.24 0l6-3.87A2.5 2.5 0 0 0 22 15.5v-7a2.5 2.5 0 0 0-.88-2.1z"/>
        <path d="M12 22V12"/>
        <path d="m21.12 6.4-9.12 5.6-9.12-5.6"/>
    </svg>''',

    # Module 1: Overview & Analytics
    "bar_chart": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="20" x2="18" y2="10"/>
        <line x1="12" y1="20" x2="12" y2="4"/>
        <line x1="6" y1="20" x2="6" y2="14"/>
        <line x1="2" y1="20" x2="22" y2="20"/>
    </svg>''',

    "dashboard": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <rect width="7" height="9" x="3" y="3" rx="1"/>
        <rect width="7" height="5" x="14" y="3" rx="1"/>
        <rect width="7" height="9" x="14" y="12" rx="1"/>
        <rect width="7" height="5" x="3" y="16" rx="1"/>
    </svg>''',

    # Module 2: Forecast Explorer & Time-Series
    "trending_up": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
        <polyline points="17 6 23 6 23 12"/>
    </svg>''',

    "sparkles": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
    </svg>''',

    # Module 2: Newsvendor & Inventory Optimization
    "scale": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
        <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
        <path d="M7 21h10"/>
        <path d="M12 3v18"/>
        <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
    </svg>''',

    # Module 3: Model Performance & Benchmarks
    "target": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <circle cx="12" cy="12" r="6"/>
        <circle cx="12" cy="12" r="2"/>
    </svg>''',

    "award": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="8" r="6"/>
        <path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>
    </svg>''',

    # Module 4: Diagnostics & Decomposition
    "microscope": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M6 18h8"/>
        <path d="M3 22h18"/>
        <path d="M14 22a7 7 0 1 0 0-14h-1"/>
        <path d="M9 14h2"/>
        <path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>
        <path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>
    </svg>''',

    "tree": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 19v3"/>
        <path d="M8 19h8"/>
        <path d="M12 2a4 4 0 0 0-4 4c0 .73.2 1.41.54 2A5 5 0 0 0 6 13a5 5 0 0 0 3 4.58V19h6v-1.42A5 5 0 0 0 18 13a5 5 0 0 0-2.54-5A4 4 0 0 0 12 2Z"/>
    </svg>''',

    # Module 5: Monitoring & MLOps Governance
    "shield_check": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <polyline points="9 12 11 14 15 10"/>
    </svg>''',

    "pulse": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>''',

    # General Utilities & Controls
    "database": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/>
        <path d="M3 5V19A9 3 0 0 0 21 19V5"/>
        <path d="M3 12A9 3 0 0 0 21 12"/>
    </svg>''',

    "cpu": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <rect width="16" height="16" x="4" y="4" rx="2"/>
        <rect width="6" height="6" x="9" y="9" rx="1"/>
        <path d="M15 2v2"/>
        <path d="M15 20v2"/>
        <path d="M2 15h2"/>
        <path d="M2 9h2"/>
        <path d="M20 15h2"/>
        <path d="M20 9h2"/>
        <path d="M9 2v2"/>
        <path d="M9 20v2"/>
    </svg>''',

    "compass": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>
    </svg>''',

    "check_circle": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>''',

    "alert_triangle": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>''',

    "search": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>''',

    "sliders": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <line x1="4" y1="21" x2="4" y2="14"/>
        <line x1="4" y1="10" x2="4" y2="3"/>
        <line x1="12" y1="21" x2="12" y2="12"/>
        <line x1="12" y1="8" x2="12" y2="3"/>
        <line x1="20" y1="21" x2="20" y2="16"/>
        <line x1="20" y1="12" x2="20" y2="3"/>
        <line x1="1" y1="14" x2="7" y2="14"/>
        <line x1="9" y1="8" x2="15" y2="8"/>
        <line x1="17" y1="16" x2="23" y2="16"/>
    </svg>''',

    "layers": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
        <polyline points="2 17 12 22 22 17"/>
        <polyline points="2 12 12 17 22 12"/>
    </svg>''',

    "table": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3v18"/>
        <rect width="18" height="18" x="3" y="3" rx="2"/>
        <path d="M3 9h18"/>
        <path d="M3 15h18"/>
    </svg>''',

    "package": '''<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
        <path d="m7.5 4.27 9 5.15"/>
        <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/>
        <path d="m3.3 7 8.7 5 8.7-5"/>
        <path d="M12 22V12"/>
    </svg>'''
}

# Color Presets (Hex + Light/Dark Tint mapping)
BADGE_COLORS = {
    "blue": {"primary": "#3b82f6", "bg": "rgba(59, 130, 246, 0.12)", "border": "rgba(59, 130, 246, 0.28)"},
    "green": {"primary": "#22c55e", "bg": "rgba(34, 197, 94, 0.12)", "border": "rgba(34, 197, 94, 0.28)"},
    "emerald": {"primary": "#10b981", "bg": "rgba(16, 185, 129, 0.12)", "border": "rgba(16, 185, 129, 0.28)"},
    "amber": {"primary": "#f59e0b", "bg": "rgba(245, 158, 11, 0.12)", "border": "rgba(245, 158, 11, 0.28)"},
    "purple": {"primary": "#a855f7", "bg": "rgba(168, 85, 247, 0.12)", "border": "rgba(168, 85, 247, 0.28)"},
    "indigo": {"primary": "#6366f1", "bg": "rgba(99, 102, 241, 0.12)", "border": "rgba(99, 102, 241, 0.28)"},
    "cyan": {"primary": "#06b6d4", "bg": "rgba(6, 182, 212, 0.12)", "border": "rgba(6, 182, 212, 0.28)"},
    "rose": {"primary": "#f43f5e", "bg": "rgba(244, 63, 94, 0.12)", "border": "rgba(244, 63, 94, 0.28)"},
    "neutral": {"primary": "#a1a1aa", "bg": "rgba(161, 161, 170, 0.10)", "border": "rgba(161, 161, 170, 0.22)"},
}

def get_svg_icon(name: str, size: int = 20, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """Returns pure SVG HTML markup for the specified icon name."""
    template = SVG_PATHS.get(name, SVG_PATHS.get("cube"))
    return template.format(size=size, color=color, stroke_width=stroke_width)

def render_icon_badge(
    icon_name: str, 
    badge_color: str = "blue", 
    size: int = 18, 
    padding: str = "8px", 
    border_radius: str = "8px"
) -> str:
    """
    Returns an HTML container rendering a modern SaaS-style duotone icon badge.
    """
    scheme = BADGE_COLORS.get(badge_color, BADGE_COLORS["blue"])
    icon_svg = get_svg_icon(icon_name, size=size, color=scheme["primary"])
    
    return f"""<div style="display:inline-flex; align-items:center; justify-content:center; padding:{padding}; background:{scheme['bg']}; border:1px solid {scheme['border']}; border-radius:{border_radius}; color:{scheme['primary']}; vertical-align:middle; line-height:1; box-shadow:0 1px 2px rgba(0,0,0,0.05);">{icon_svg}</div>"""

def render_section_header(
    title: str, 
    icon_name: str = "bar_chart", 
    badge_color: str = "blue", 
    subtitle: Optional[str] = None, 
    heading_level: int = 3
):
    """
    Renders a unified section header with a modern vector badge, typography, and optional subtitle.
    """
    badge_html = render_icon_badge(icon_name, badge_color=badge_color, size=18, padding="7px")
    sub_html = f'<div style="font-size:0.84rem; color:var(--text-muted); margin-top:0.25rem; font-weight:400;">{subtitle}</div>' if subtitle else ""
    
    font_size = "1.25rem" if heading_level == 3 else ("1.45rem" if heading_level == 2 else "1.1rem")
    
    html = f"""
    <div style="margin-top:1.25rem; margin-bottom:0.9rem;">
        <div style="display:flex; align-items:center; gap:0.65rem;">
            {badge_html}
            <span style="font-size:{font_size}; font-weight:700; letter-spacing:-0.02em; color:var(--text);">{title}</span>
        </div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
