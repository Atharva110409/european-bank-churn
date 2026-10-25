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
# 1. COLOR PALETTE CONSTANTS (SOVEREIGN AUDIT GOLD & ZINC NEUTRALS)
# -----------------------------------------------------------------------------
# Primary brand & interaction accent: Sovereign Audit Gold / Warm Amber
COLOR_PRIMARY = "#f59e0b"
COLOR_PRIMARY_LIGHT = "#fbbf24"
COLOR_PRIMARY_DARK = "#d97706"
COLOR_PRIMARY_BG = "rgba(245, 158, 11, 0.12)"

# Unified Accent: Warm Sovereign Gold
COLOR_ACCENT = "#f59e0b"
COLOR_ACCENT_LIGHT = "#fbbf24"
COLOR_ACCENT_DARK = "#d97706"
COLOR_ACCENT_BG = "rgba(245, 158, 11, 0.12)"

# Semantic Pair: Retained (Stability/Positive) vs Exited (Risk/Churn)
# Colorblind-safe pair: Teal (#0d9488) vs Coral/Rose (#f43f5e) - STRICTLY RESERVED FOR DATA ONLY
COLOR_RETAINED = "#0d9488"
COLOR_RETAINED_LIGHT = "#14b8a6"
COLOR_RETAINED_BG = "rgba(13, 148, 136, 0.14)"

COLOR_EXITED = "#f43f5e"
COLOR_EXITED_LIGHT = "#fb7185"
COLOR_EXITED_BG = "rgba(244, 63, 94, 0.14)"

# Functional / Alert Colors
COLOR_WARNING = "#f59e0b"         # Amber / Caution / Small Samples
COLOR_WARNING_BG = "rgba(245, 158, 11, 0.14)"

# Consolidated Info & Attribution (Unified with Sovereign Gold, retiring separate indigo/purple)
COLOR_INFO = "#f59e0b"
COLOR_INFO_BG = "rgba(245, 158, 11, 0.12)"

COLOR_PURPLE = "#f59e0b"          # Unified to amber/gold to consolidate palette
COLOR_PURPLE_BG = "rgba(245, 158, 11, 0.12)"

# Neutral Grays (Zinc True Neutrals)
COLOR_NEUTRAL = "#71717a"         # Neutral zinc for baseline references
COLOR_NEUTRAL_LIGHT = "#a1a1aa"
COLOR_NEUTRAL_DARK = "#27272a"
COLOR_NEUTRAL_BG = "rgba(255, 255, 255, 0.08)"

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
# BESPOKE INLINE SVG LOGO MARK (BANK FACADE + DIAGNOSTIC VECTOR)
# -----------------------------------------------------------------------------
LOGO_SVG = """<svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; flex-shrink: 0;">
  <path d="M3 9.5L14 3.5L25 9.5H3Z" fill="#f59e0b" fill-opacity="0.95"/>
  <rect x="4" y="10" width="20" height="2" rx="0.5" fill="#a1a1aa"/>
  <rect x="5.5" y="12.5" width="2.5" height="8.5" rx="0.5" fill="#d4d4d8"/>
  <rect x="10.5" y="12.5" width="2.5" height="8.5" rx="0.5" fill="#d4d4d8"/>
  <rect x="15" y="12.5" width="2.5" height="8.5" rx="0.5" fill="#d4d4d8"/>
  <rect x="20" y="12.5" width="2.5" height="8.5" rx="0.5" fill="#d4d4d8"/>
  <rect x="3" y="21.5" width="22" height="2.5" rx="0.5" fill="#a1a1aa"/>
  <path d="M7 8.5L15 15.5L22 20.5" stroke="#f43f5e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="22" cy="20.5" r="2.2" fill="#f43f5e"/>
</svg>"""


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
        --eb-bg-card: rgba(255, 255, 255, 0.94);
        --eb-bg-card-subtle: rgba(248, 250, 252, 0.90);
        --eb-bg-hover: rgba(241, 245, 249, 0.95);
        --eb-border-card: rgba(148, 163, 184, 0.28);
        --eb-border-hover: rgba(37, 99, 235, 0.45);
        --eb-shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.05);
        --eb-shadow-md: 0 4px 16px rgba(15, 23, 42, 0.06);
        --eb-shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.08);
        
        --eb-text-title: #0f172a;
        --eb-text-body: #334155;
        --eb-text-label: #64748b;
        --eb-text-subtle: #94a3b8;
        
        --eb-gridline: rgba(148, 163, 184, 0.18);
        --eb-badge-bg: rgba(217, 119, 6, 0.12);
        --eb-badge-text: #b45309;
    }

    /* Dark Mode (System preference or Streamlit dark theme) - True Black / Near-Black */
    @media (prefers-color-scheme: dark) {
        :root {
            --eb-bg-card: rgba(22, 22, 24, 0.94);
            --eb-bg-card-subtle: rgba(28, 28, 32, 0.88);
            --eb-bg-hover: rgba(38, 38, 44, 0.95);
            --eb-border-card: rgba(255, 255, 255, 0.12);
            --eb-border-hover: rgba(245, 158, 11, 0.55);
            --eb-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.6);
            --eb-shadow-md: 0 4px 20px rgba(0, 0, 0, 0.75);
            --eb-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.9);
            
            --eb-text-title: #f5f5f5;
            --eb-text-body: #d4d4d8;
            --eb-text-label: #a1a1aa;
            --eb-text-subtle: #71717a;
            
            --eb-gridline: rgba(255, 255, 255, 0.09);
            --eb-badge-bg: rgba(245, 158, 11, 0.18);
            --eb-badge-text: #fbbf24;
        }
    }

    /* Streamlit explicit dark theme overrides - True Black / Near-Black */
    [data-theme="dark"], .stApp[data-theme="dark"] {
        --eb-bg-card: rgba(22, 22, 24, 0.94);
        --eb-bg-card-subtle: rgba(28, 28, 32, 0.88);
        --eb-bg-hover: rgba(38, 38, 44, 0.95);
        --eb-border-card: rgba(255, 255, 255, 0.12);
        --eb-border-hover: rgba(245, 158, 11, 0.55);
        --eb-shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.6);
        --eb-shadow-md: 0 4px 20px rgba(0, 0, 0, 0.75);
        --eb-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.9);
        
        --eb-text-title: #f5f5f5;
        --eb-text-body: #d4d4d8;
        --eb-text-label: #a1a1aa;
        --eb-text-subtle: #71717a;
        --eb-gridline: rgba(255, 255, 255, 0.09);
        --eb-badge-bg: rgba(245, 158, 11, 0.18);
        --eb-badge-text: #fbbf24;
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
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(22, 22, 24, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 5px solid #f59e0b;
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
        background: rgba(245, 158, 11, 0.22);
        color: #fbbf24;
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
        border-top: 1px solid rgba(245, 158, 11, 0.25);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .callout-disclaimer strong {
        color: #fbbf24;
    }
    .callout-tradeoff {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left: 4px solid #f59e0b;
        border-top: 1px solid rgba(245, 158, 11, 0.25);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .callout-tradeoff strong {
        color: #fbbf24;
    }
    .callout-methodology {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.09) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-left: 4px solid #d97706;
        border-top: 1px solid rgba(217, 119, 6, 0.25);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .callout-methodology strong {
        color: #fbbf24;
    }
    .callout-causal {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.07) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-left: 4px solid #b45309;
        border-top: 1px solid rgba(180, 83, 9, 0.25);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .callout-causal strong {
        color: #fbbf24;
    }

    /* Small Sample Flag Banner */
    .small-sample-warning {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.825rem;
        color: var(--eb-text-body);
        line-height: 1.45;
    }
    .small-sample-warning strong {
        color: #fbbf24;
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
        background: rgba(245, 158, 11, 0.16);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-navy {
        background: rgba(245, 158, 11, 0.16);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    /* Sidebar Branding with Bespoke SVG */
    .sidebar-brand-card {
        padding: 0.95rem 1.1rem;
        border-radius: 12px;
        background: var(--eb-bg-card-subtle);
        border: 1px solid var(--eb-border-card);
        margin-bottom: 1.25rem;
        box-shadow: var(--eb-shadow-sm);
    }
    .sidebar-brand-title {
        font-family: var(--eb-font-heading);
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--eb-text-title);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .sidebar-brand-subtitle {
        font-size: 0.725rem;
        font-weight: 500;
        color: var(--eb-text-label);
        margin-top: 0.3rem;
        letter-spacing: 0.02em;
    }

    /* Cohort Demographic Risk Heat Strip Component */
    .risk-heat-strip-card {
        background: var(--eb-bg-card);
        border: 1px solid var(--eb-border-card);
        border-radius: 14px;
        padding: 1.15rem 1.4rem;
        box-shadow: var(--eb-shadow-md);
        margin-bottom: 1.4rem;
    }
    .risk-strip-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .risk-strip-title {
        font-family: var(--eb-font-heading);
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--eb-text-title);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    .risk-strip-baseline {
        font-size: 0.775rem;
        color: var(--eb-text-label);
        background: var(--eb-bg-card-subtle);
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid var(--eb-border-card);
    }
    .risk-strip-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
    }
    @media (max-width: 900px) {
        .risk-strip-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .risk-cohort-item {
        background: var(--eb-bg-card-subtle);
        border: 1px solid var(--eb-border-card);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .risk-cohort-item:hover {
        transform: translateY(-2px);
        border-color: var(--eb-border-hover);
    }
    .risk-cohort-item.spike-alert {
        border: 1.5px solid rgba(244, 63, 94, 0.5);
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.08) 0%, rgba(217, 119, 6, 0.08) 100%);
    }
    .risk-cohort-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--eb-text-label);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .risk-cohort-rate {
        font-family: var(--eb-font-heading);
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--eb-text-title);
        line-height: 1.15;
        margin: 0.35rem 0 0.2rem 0;
    }
    .risk-cohort-rate.spike {
        color: #f43f5e;
    }
    .risk-cohort-sub {
        font-size: 0.725rem;
        color: var(--eb-text-subtle);
        margin-bottom: 0.55rem;
    }
    .risk-bar-track {
        height: 6px;
        background: rgba(148, 163, 184, 0.2);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 999px;
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
    icon: Optional[str] = None,
    title: str = "AuditBank Europe",
    subtitle: str = "Leakage-Safe Statistical Audit & Generalization Benchmark",
    badge: Optional[str] = "Audited Analytics"
) -> None:
    """
    Renders an elevated dashboard hero header with custom SVG/icon, title, subtitle, and status pill.
    """
    display_icon = icon if icon else LOGO_SVG
    badge_html = f'<span class="badge badge-info">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="page-header-card">
        <div class="page-header-left">
            <div class="page-header-icon">{display_icon}</div>
            <div>
                <h1 class="page-header-title">{title}</h1>
                <div class="page-header-subtitle">{subtitle}</div>
            </div>
        </div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_branding(
    app_title: str = "AuditBank Europe",
    app_subtitle: str = "Leakage-Safe Churn Audit • v1.0"
) -> None:
    """
    Renders a unified top brand card in the sidebar navigation with custom SVG branding.
    """
    st.sidebar.markdown(f"""
    <div class="sidebar-brand-card">
        <div class="sidebar-brand-title">{LOGO_SVG} <span>{app_title}</span></div>
        <div class="sidebar-brand-subtitle">{app_subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_cohort_risk_heat_strip(
    age_summary: Any,
    overall_churn_rate: float = 0.2037
) -> None:
    """
    Renders an executive demographic risk heat strip comparing all 4 age cohorts side-by-side.
    Visually showcases the 46-59 churn concentration spike against the portfolio baseline.
    """
    items_html = []
    for _, row in age_summary.iterrows():
        seg = str(row["segment"])
        n_val = int(row["n"])
        rate_val = float(row["churn_rate"])
        exits_val = int(row["exited"])
        multiplier = rate_val / overall_churn_rate if overall_churn_rate > 0 else 1.0

        is_spike = (seg == "46-59")
        item_class = "risk-cohort-item spike-alert" if is_spike else "risk-cohort-item"
        rate_class = "risk-cohort-rate spike" if is_spike else "risk-cohort-rate"

        if is_spike:
            tag_html = f'<span class="kpi-delta-pill delta-exited" style="font-size: 0.65rem; padding: 0.15rem 0.45rem;">🔥 {multiplier:.2f}x Baseline</span>'
            fill_style = f"width: {min(rate_val * 100, 100):.1f}%; background: linear-gradient(90deg, #f59e0b 0%, #f43f5e 100%); box-shadow: 0 0 10px rgba(244, 63, 94, 0.4);"
        elif rate_val < overall_churn_rate:
            tag_html = '<span class="kpi-delta-pill delta-retained" style="font-size: 0.65rem; padding: 0.15rem 0.45rem;">Low Risk</span>'
            fill_style = f"width: {min(rate_val * 100, 100):.1f}%; background: #0d9488;"
        else:
            tag_html = '<span class="kpi-delta-pill delta-warning" style="font-size: 0.65rem; padding: 0.15rem 0.45rem;">Elevated</span>'
            fill_style = f"width: {min(rate_val * 100, 100):.1f}%; background: #f59e0b;"

        items_html.append(
            f'<div class="{item_class}">'
            f'<div class="risk-cohort-label"><span>Age {seg}</span>{tag_html}</div>'
            f'<div class="{rate_class}">{rate_val:.1%}</div>'
            f'<div class="risk-cohort-sub">{exits_val:,} exits / {n_val:,} accounts</div>'
            f'<div class="risk-bar-track"><div class="risk-bar-fill" style="{fill_style}"></div></div>'
            f'</div>'
        )

    grid_content = "".join(items_html)

    html = (
        f'<div class="risk-heat-strip-card">'
        f'<div class="risk-strip-header">'
        f'<div class="risk-strip-title"><span>📊</span> Demographic Churn Risk Heat Strip (Empirical Age Cohorts)</div>'
        f'<div class="risk-strip-baseline">Portfolio Baseline: <strong>{overall_churn_rate:.2%} Churn</strong></div>'
        f'</div>'
        f'<div class="risk-strip-grid">{grid_content}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
            f'<div style="background: rgba(245, 158, 11, 0.16); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 10px; padding: 0.75rem 1.25rem; text-align: center; min-width: 140px;">'
            f'<div style="font-family: var(--eb-font-heading); font-size: 1.6rem; font-weight: 800; color: #fbbf24; line-height: 1.1;">{stat_value}</div>'
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
            color="#d4d4d8",
            size=12
        ),
        showlegend=show_legend,
        height=height,
        margin=dict(t=35 if title else 20, b=30, l=35, r=25),
        hoverlabel=dict(
            bgcolor="rgba(22, 22, 24, 0.96)",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="rgba(245, 158, 11, 0.4)",
            font=dict(color="#f5f5f5")
        )
    )

    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(
                    family="Outfit, sans-serif",
                    size=14,
                    color="#f5f5f5"
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
        tickfont=dict(size=11, color="#a1a1aa")
    )
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.10)",
        zeroline=False,
        tickfont=dict(size=11, color="#a1a1aa")
    )

    return fig
