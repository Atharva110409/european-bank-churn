"""
ui_theme.py
Design System & Reusable UI Components for European Bank Churn Analytics.

Provides:
- Color palette constants (primary accent, colorblind-safe retained vs exited semantic pair, neutrals).
- Dual theme support (light and dark mode matching viewer's system/browser preference).
- Reusable KPI cards with delta pills, icons, and micro-hover states.
- Reusable page headers, sidebar branding, and grouped content cards.
- Structured callout boxes (provenance disclaimer, methodology, threshold tradeoff, causal notices).
- Consistent Plotly chart styler enforcing shared color mappings, clean typography, and subtle gridlines.
"""

from typing import Optional, Dict, Any
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. COLOR PALETTE CONSTANTS
# -----------------------------------------------------------------------------
# Primary brand & interaction accent (Modern Indigo)
COLOR_PRIMARY = "#6366f1"
COLOR_PRIMARY_LIGHT = "#818cf8"
COLOR_PRIMARY_DARK = "#4f46e5"
COLOR_PRIMARY_BG = "rgba(99, 102, 241, 0.10)"

# Semantic Pair: Retained (Stability/Positive) vs Exited (Risk/Churn)
# Colorblind-safe pair: Teal (#0d9488) vs Coral/Rose (#f43f5e)
COLOR_RETAINED = "#0d9488"
COLOR_RETAINED_LIGHT = "#14b8a6"
COLOR_RETAINED_BG = "rgba(13, 148, 136, 0.12)"

COLOR_EXITED = "#f43f5e"
COLOR_EXITED_LIGHT = "#fb7185"
COLOR_EXITED_BG = "rgba(244, 63, 94, 0.12)"

# Functional / Alert Colors
COLOR_WARNING = "#f59e0b"         # Amber / Caution / Small Samples
COLOR_WARNING_BG = "rgba(245, 158, 11, 0.12)"

COLOR_INFO = "#3b82f6"            # Sky Blue / Technical Details
COLOR_INFO_BG = "rgba(59, 130, 246, 0.10)"

COLOR_PURPLE = "#8b5cf6"          # Purple / Explainability / Causal
COLOR_PURPLE_BG = "rgba(139, 92, 246, 0.10)"

# Neutral Grays
COLOR_NEUTRAL = "#64748b"         # Slate neutral for baseline references
COLOR_NEUTRAL_LIGHT = "#94a3b8"
COLOR_NEUTRAL_DARK = "#334155"
COLOR_NEUTRAL_BG = "rgba(100, 116, 139, 0.10)"

# Standard Color Mapping for Charts
CHART_COLOR_MAP: Dict[str, str] = {
    "Retained": COLOR_RETAINED,
    "Retained (0)": COLOR_RETAINED,
    "0": COLOR_RETAINED,
    "Exited": COLOR_EXITED,
    "Exited (1)": COLOR_EXITED,
    "1": COLOR_EXITED,
}


# -----------------------------------------------------------------------------
# 2. GLOBAL CSS THEME INJECTION (LIGHT & DARK MODE AWARE)
# -----------------------------------------------------------------------------
def apply_custom_theme() -> None:
    """
    Applies custom CSS variables and utility classes.
    Automatically adapts to the viewer's system light or dark mode theme.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* CSS Design System Tokens */
    :root {
        --eb-font-body: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --eb-font-heading: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        
        /* Light Mode Defaults */
        --eb-bg-card: rgba(255, 255, 255, 0.92);
        --eb-bg-card-subtle: rgba(248, 250, 252, 0.85);
        --eb-bg-hover: rgba(241, 245, 249, 0.95);
        --eb-border-card: rgba(148, 163, 184, 0.25);
        --eb-border-hover: rgba(99, 102, 241, 0.45);
        --eb-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.05);
        --eb-shadow-md: 0 4px 16px rgba(15, 23, 42, 0.06);
        --eb-shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.08);
        
        --eb-text-title: #0f172a;
        --eb-text-body: #334155;
        --eb-text-label: #64748b;
        --eb-text-subtle: #94a3b8;
        
        --eb-gridline: rgba(148, 163, 184, 0.18);
        --eb-badge-bg: rgba(99, 102, 241, 0.10);
        --eb-badge-text: #4f46e5;
    }

    /* Dark Mode (System preference or Streamlit dark theme) */
    @media (prefers-color-scheme: dark) {
        :root {
            --eb-bg-card: linear-gradient(135deg, rgba(30, 41, 59, 0.72) 0%, rgba(15, 23, 42, 0.82) 100%);
            --eb-bg-card-subtle: rgba(30, 41, 59, 0.55);
            --eb-bg-hover: rgba(51, 65, 85, 0.65);
            --eb-border-card: rgba(255, 255, 255, 0.08);
            --eb-border-hover: rgba(99, 102, 241, 0.5);
            --eb-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.2);
            --eb-shadow-md: 0 4px 20px rgba(0, 0, 0, 0.32);
            --eb-shadow-lg: 0 8px 30px rgba(0, 0, 0, 0.45);
            
            --eb-text-title: #f8fafc;
            --eb-text-body: #cbd5e1;
            --eb-text-label: #94a3b8;
            --eb-text-subtle: #64748b;
            
            --eb-gridline: rgba(255, 255, 255, 0.08);
            --eb-badge-bg: rgba(99, 102, 241, 0.20);
            --eb-badge-text: #a5b4fc;
        }
    }

    /* Streamlit explicit dark theme overrides */
    [data-theme="dark"], .stApp[data-theme="dark"] {
        --eb-bg-card: linear-gradient(135deg, rgba(30, 41, 59, 0.72) 0%, rgba(15, 23, 42, 0.82) 100%);
        --eb-bg-card-subtle: rgba(30, 41, 59, 0.55);
        --eb-border-card: rgba(255, 255, 255, 0.08);
        --eb-border-hover: rgba(99, 102, 241, 0.5);
        --eb-text-title: #f8fafc;
        --eb-text-body: #cbd5e1;
        --eb-text-label: #94a3b8;
        --eb-text-subtle: #64748b;
        --eb-gridline: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: var(--eb-font-body);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--eb-font-heading);
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--eb-text-title);
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3.5rem;
        max-width: 1280px;
    }

    /* Header Banner Component */
    .page-header-card {
        background: var(--eb-bg-card);
        border: 1px solid var(--eb-border-card);
        border-radius: 14px;
        padding: 1.25rem 1.6rem;
        box-shadow: var(--eb-shadow-sm);
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.25rem;
    }
    .page-header-left {
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }
    .page-header-icon {
        font-size: 2.2rem;
        line-height: 1;
        padding: 0.75rem;
        border-radius: 12px;
        background: var(--eb-badge-bg);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .page-header-title {
        font-family: var(--eb-font-heading);
        font-size: 1.65rem;
        font-weight: 800;
        color: var(--eb-text-title);
        margin: 0;
        line-height: 1.2;
    }
    .page-header-subtitle {
        font-size: 0.875rem;
        color: var(--eb-text-label);
        margin-top: 0.2rem;
    }

    /* Reusable KPI Card */
    .kpi-card {
        background: var(--eb-bg-card);
        border: 1px solid var(--eb-border-card);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        box-shadow: var(--eb-shadow-md);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--eb-border-hover);
        box-shadow: var(--eb-shadow-lg);
    }
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--eb-text-label);
    }
    .kpi-icon {
        font-size: 1rem;
        opacity: 0.75;
    }
    .kpi-value {
        font-family: var(--eb-font-heading);
        font-size: 1.95rem;
        font-weight: 800;
        color: var(--eb-text-title);
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    .kpi-subtext {
        font-size: 0.775rem;
        color: var(--eb-text-label);
        margin-top: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .kpi-delta-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.725rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .delta-retained {
        background: rgba(13, 148, 136, 0.15);
        color: #0d9488;
    }
    .delta-exited {
        background: rgba(244, 63, 94, 0.15);
        color: #f43f5e;
    }
    .delta-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
    }
    .delta-neutral {
        background: var(--eb-badge-bg);
        color: var(--eb-badge-text);
    }

    /* Content Card Container */
    .content-card {
        background: var(--eb-bg-card);
        border: 1px solid var(--eb-border-card);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: var(--eb-shadow-md);
        backdrop-filter: blur(10px);
        margin-bottom: 1.25rem;
    }
    .content-card-title {
        font-family: var(--eb-font-heading);
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--eb-text-title);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Headline Strategic Callout */
    .headline-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.10) 0%, rgba(13, 148, 136, 0.06) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        box-shadow: var(--eb-shadow-md);
        margin-bottom: 1.25rem;
    }
    .headline-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        margin-bottom: 0.6rem;
    }
    .headline-text {
        font-size: 1.1rem;
        color: var(--eb-text-title);
        line-height: 1.55;
    }

    /* Standardized Callouts */
    .callout-box {
        border-radius: 0 10px 10px 0;
        padding: 0.95rem 1.35rem;
        margin-bottom: 1.25rem;
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--eb-text-body);
        backdrop-filter: blur(8px);
    }
    .callout-box strong {
        font-weight: 700;
    }
    .callout-disclaimer {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.10) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left: 4px solid #f59e0b;
        border-top: 1px solid rgba(245, 158, 11, 0.2);
        border-right: 1px solid rgba(245, 158, 11, 0.12);
        border-bottom: 1px solid rgba(245, 158, 11, 0.12);
    }
    .callout-disclaimer strong {
        color: #f59e0b;
    }
    .callout-tradeoff {
        background: linear-gradient(90deg, rgba(249, 115, 22, 0.10) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left: 4px solid #f97316;
        border-top: 1px solid rgba(249, 115, 22, 0.2);
        border-right: 1px solid rgba(249, 115, 22, 0.12);
        border-bottom: 1px solid rgba(249, 115, 22, 0.12);
    }
    .callout-tradeoff strong {
        color: #f97316;
    }
    .callout-methodology {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.10) 0%, rgba(59, 130, 246, 0.04) 100%);
        border-left: 4px solid #6366f1;
        border-top: 1px solid rgba(99, 102, 241, 0.2);
        border-right: 1px solid rgba(99, 102, 241, 0.12);
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
    }
    .callout-methodology strong {
        color: #818cf8;
    }
    .callout-causal {
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.10) 0%, rgba(99, 102, 241, 0.04) 100%);
        border-left: 4px solid #8b5cf6;
        border-top: 1px solid rgba(139, 92, 246, 0.2);
        border-right: 1px solid rgba(139, 92, 246, 0.12);
        border-bottom: 1px solid rgba(139, 92, 246, 0.12);
    }
    .callout-causal strong {
        color: #a78bfa;
    }

    /* Small Sample Flag Banner */
    .small-sample-warning {
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.825rem;
        color: var(--eb-text-body);
        line-height: 1.45;
    }
    .small-sample-warning strong {
        color: #f59e0b;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-winner {
        background: rgba(13, 148, 136, 0.18);
        color: #0d9488;
        border: 1px solid rgba(13, 148, 136, 0.4);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.18);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-danger {
        background: rgba(244, 63, 94, 0.18);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .badge-info {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Sidebar Branding */
    .sidebar-brand-card {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        background: var(--eb-bg-card-subtle);
        border: 1px solid var(--eb-border-card);
        margin-bottom: 1.25rem;
    }
    .sidebar-brand-title {
        font-family: var(--eb-font-heading);
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--eb-text-title);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-brand-subtitle {
        font-size: 0.75rem;
        color: var(--eb-text-label);
        margin-top: 0.2rem;
    }

    /* Streamlit DataFrame Container Polish */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--eb-border-card);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. REUSABLE UI COMPONENTS
# -----------------------------------------------------------------------------
def render_page_header(
    icon: str,
    title: str,
    subtitle: str,
    badge: Optional[str] = "Audited Analytics"
) -> None:
    """
    Renders an elevated dashboard hero header with icon, title, subtitle, and status pill.
    """
    badge_html = f'<span class="badge badge-info">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="page-header-card">
        <div class="page-header-left">
            <div class="page-header-icon">{icon}</div>
            <div>
                <h1 class="page-header-title">{title}</h1>
                <div class="page-header-subtitle">{subtitle}</div>
            </div>
        </div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_branding() -> None:
    """
    Renders a unified top brand card in the sidebar navigation.
    """
    st.sidebar.markdown("""
    <div class="sidebar-brand-card">
        <div class="sidebar-brand-title">🏦 European Bank Churn</div>
        <div class="sidebar-brand-subtitle">Executive Risk & Portfolio Analytics • v1.0</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    delta: str = "",
    delta_type: str = "neutral",
    icon: str = ""
) -> None:
    """
    Renders a stylized glassmorphism KPI card.
    Supports optional delta badge and icons.
    """
    delta_class = f"delta-{delta_type}" if delta_type in ["retained", "exited", "warning", "neutral"] else "delta-neutral"
    delta_html = f'<span class="kpi-delta-pill {delta_class}">{delta}</span>' if delta else ""
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""

    subtext_content = ""
    if delta or subtext:
        subtext_content = f'<div class="kpi-subtext">{delta_html}<span>{subtext}</span></div>'

    html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-header"><span class="kpi-label">{label}</span>{icon_html}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{subtext_content}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_headline_callout(
    title: str,
    text: str,
    stat_value: Optional[str] = None,
    stat_label: Optional[str] = None,
    badge: str = "Key Strategic Finding"
) -> None:
    """
    Renders a high-impact executive strategic finding banner.
    Specifically styled for the Age 46-59 churn concentration finding.
    """
    stat_html = ""
    if stat_value and stat_label:
        stat_html = (
            f'<div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 0.75rem 1.25rem; text-align: center; min-width: 140px;">'
            f'<div style="font-family: var(--eb-font-heading); font-size: 1.6rem; font-weight: 800; color: #818cf8; line-height: 1.1;">{stat_value}</div>'
            f'<div style="font-size: 0.725rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--eb-text-label); margin-top: 0.2rem;">{stat_label}</div>'
            f'</div>'
        )

    html = (
        f'<div class="headline-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1.5rem; flex-wrap: wrap;">'
        f'<div style="flex: 1; min-width: 280px;">'
        f'<span class="headline-badge">🔍 {badge}</span>'
        f'<div style="font-family: var(--eb-font-heading); font-size: 1.25rem; font-weight: 700; color: var(--eb-text-title); margin-bottom: 0.45rem;">{title}</div>'
        f'<div class="headline-text">{text}</div>'
        f'</div>'
        f'{stat_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_callout(
    title: str,
    text: str,
    callout_type: str = "info",
    icon: str = ""
) -> None:
    """
    Renders a structured callout container with specific semantic coloring.
    Valid callout_types: 'disclaimer', 'tradeoff', 'methodology', 'causal', 'warning', 'info'.
    """
    type_class_map = {
        "disclaimer": "callout-disclaimer",
        "tradeoff": "callout-tradeoff",
        "methodology": "callout-methodology",
        "causal": "callout-causal",
        "warning": "small-sample-warning",
        "info": "callout-methodology"
    }
    box_class = type_class_map.get(callout_type, "callout-methodology")
    icon_prefix = f"{icon} " if icon else ""

    html = (
        f'<div class="callout-box {box_class}">'
        f'<strong>{icon_prefix}{title}</strong><br>{text}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_provenance_disclaimer() -> None:
    """
    Renders the mandatory dataset provenance and limitation statement.
    """
    render_callout(
        title="DATA PROVENANCE & METHODOLOGICAL BOUNDARY STATEMENT:",
        text="""This dashboard presents retrospective exploratory analytics on an educational customer-churn benchmark dataset 
        supplied for this project. The exact provenance has not been independently verified. 
        <strong>Do not present or interpret this data as actual European Central Bank (ECB) records or proprietary data from any commercial European bank.</strong> 
        The findings represent descriptive statistical associations and retrospective risk ranking — they do not demonstrate causality, 
        forecast churn timing, represent bank-wide operations, or estimate revenue loss.""",
        callout_type="disclaimer",
        icon="⚖️"
    )


def render_badge(text: str, badge_type: str = "info") -> str:
    """
    Helper returning an inline HTML badge tag string.
    """
    return f'<span class="badge badge-{badge_type}">{text}</span>'


# -----------------------------------------------------------------------------
# 4. CONSISTENT PLOTLY CHART STYLER
# -----------------------------------------------------------------------------
def style_plotly_figure(
    fig: go.Figure,
    height: int = 320,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    show_legend: bool = True
) -> go.Figure:
    """
    Applies unified financial-dashboard styling to any Plotly figure:
    - Transparent paper & plot backgrounds
    - Consistent typography ('Inter', sans-serif)
    - Subtle, non-dominating gridlines (10% opacity)
    - Clean hoverlabel styling and balanced margins
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#94a3b8",
            size=12
        ),
        showlegend=show_legend,
        height=height,
        margin=dict(t=35 if title else 20, b=30, l=35, r=25),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="rgba(255, 255, 255, 0.15)"
        )
    )

    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(
                    family="Outfit, sans-serif",
                    size=14,
                    color="#cbd5e1"
                ),
                x=0.01,
                xanchor="left"
            )
        )

    # Style axes if present
    fig.update_xaxes(
        title=xaxis_title,
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=11, color="#94a3b8")
    )
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
        tickfont=dict(size=11, color="#94a3b8")
    )

    return fig
