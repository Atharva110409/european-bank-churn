"""
pages/1_Segments.py
Customer Segmentation Analytics: Country, Age Band, Product Count, and Activity.

Features interactive filtering, handles zero-result filter states gracefully,
displays sample sizes (n) beside every rate, and visually flags small-sample
product segments (n < 300).
All numbers trace directly to data validation audit and live segmentation routines.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_validation import load_and_validate_data
from src.segmentation import (
    summarize_geography,
    summarize_age_bands,
    summarize_num_of_products,
    summarize_is_active_member
)
from src.ui_theme import (
    apply_custom_theme,
    render_page_header,
    render_sidebar_branding,
    render_provenance_disclaimer,
    render_kpi_card,
    render_callout,
    style_plotly_figure,
    COLOR_PRIMARY,
    COLOR_RETAINED,
    COLOR_EXITED,
    COLOR_WARNING,
    COLOR_NEUTRAL
)

st.set_page_config(
    page_title="Customer Segments | European Bank Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_theme()

# Unified sidebar brand header
render_sidebar_branding()

# Top Hero Page Header
render_page_header(
    icon="📊",
    title="Customer Segmentation Analytics",
    subtitle="Cohort Risk Analysis Across Geography, Age Demographics, Product Depth, and Engagement",
    badge="Segment Intelligence"
)

# Mandatory Provenance Disclaimer Callout
render_provenance_disclaimer()

# Load data
@st.cache_data
def get_data():
    return load_and_validate_data("data/raw/European_Bank.csv")

try:
    df_raw, audit = get_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Total baseline exits for calculating share of all exits
portfolio_total_exits = audit["total_exited"]
portfolio_total_customers = audit["row_count"]
portfolio_baseline_churn = portfolio_total_exits / portfolio_total_customers

# ----------------- Interactive Filters in Sidebar -----------------
st.sidebar.markdown("### 🔍 Filter Customer Cohort")

st.sidebar.markdown("**Demographic Profiles**")
all_geos = sorted(df_raw["Geography"].unique().tolist())
selected_geos = st.sidebar.multiselect("Geography", all_geos, default=all_geos)

all_genders = sorted(df_raw["Gender"].unique().tolist())
selected_genders = st.sidebar.multiselect("Gender", all_genders, default=all_genders)

st.sidebar.markdown("---")
st.sidebar.markdown("**Financial & Engagement Behavior**")
card_filter = st.sidebar.selectbox("Credit Card Ownership", ["All Customers", "Has Card (1)", "No Card (0)"])
active_filter = st.sidebar.selectbox("Platform Activity", ["All Customers", "Active Member (1)", "Inactive Member (0)"])

# Filter dataframe
filtered_df = df_raw.copy()

if selected_geos:
    filtered_df = filtered_df[filtered_df["Geography"].isin(selected_geos)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_genders:
    filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
else:
    filtered_df = filtered_df.iloc[0:0]

if card_filter == "Has Card (1)":
    filtered_df = filtered_df[filtered_df["HasCrCard"] == 1]
elif card_filter == "No Card (0)":
    filtered_df = filtered_df[filtered_df["HasCrCard"] == 0]

if active_filter == "Active Member (1)":
    filtered_df = filtered_df[filtered_df["IsActiveMember"] == 1]
elif active_filter == "Inactive Member (0)":
    filtered_df = filtered_df[filtered_df["IsActiveMember"] == 0]

filtered_n = len(filtered_df)
filtered_exits = int(filtered_df["Exited"].sum()) if filtered_n > 0 else 0
filtered_churn = filtered_exits / filtered_n if filtered_n > 0 else 0.0

# ----------------- Zero-Result Handling -----------------
if filtered_n == 0:
    render_callout(
        title="ZERO RECORDS MATCH FILTER CRITERIA",
        text="No customer records meet all current filter selections. Please expand your selections in the sidebar (e.g., select at least one Country or Gender).",
        callout_type="warning",
        icon="⚠️"
    )
    st.stop()

# ----------------- Filter Cohort KPI Bar -----------------
st.markdown("### 🎯 Active Filtered Cohort Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_kpi_card(
        label="Filtered Cohort Size",
        value=f"{filtered_n:,}",
        subtext="of total file records",
        delta=f"{(filtered_n / portfolio_total_customers):.1%} of portfolio",
        delta_type="neutral",
        icon="👥"
    )
with kpi2:
    render_kpi_card(
        label="Observed Cohort Exits",
        value=f"{filtered_exits:,}",
        subtext="in selected cohort",
        delta=f"{filtered_churn:.2%} cohort churn",
        delta_type="exited" if filtered_churn > portfolio_baseline_churn else "retained",
        icon="📉"
    )
with kpi3:
    render_kpi_card(
        label="Retained In Cohort",
        value=f"{(filtered_n - filtered_exits):,}",
        subtext="active customers",
        delta=f"{(1 - filtered_churn):.2%} retention",
        delta_type="retained",
        icon="🛡️"
    )
with kpi4:
    render_kpi_card(
        label="Share of Bank Exits",
        value=f"{(filtered_exits / portfolio_total_exits):.1%}",
        subtext=f"{filtered_exits:,} of {portfolio_total_exits:,} exits",
        delta="Portfolio Exit Share",
        delta_type="warning" if (filtered_exits / portfolio_total_exits) > 0.4 else "neutral",
        icon="🏦"
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# Helper function to render segment section in unified cards
def render_segment_section(
    title: str,
    subtitle: str,
    table_df: pd.DataFrame,
    flag_small_sample: bool = False,
    section_icon: str = "📌"
):
    st.markdown(f"""
    <div class="content-card">
        <div class="content-card-title">{section_icon} {title}</div>
        <div style="font-size: 0.8rem; color: var(--eb-text-label); margin-top: -0.4rem; margin-bottom: 0.85rem;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_table, col_chart = st.columns([1.1, 1.25])
    
    with col_table:
        display_df = table_df.copy()
        
        # Display sample size (n) clearly beside every rate
        display_df["Churn Rate (n)"] = display_df.apply(
            lambda r: f"{r['churn_rate']:.2%} (n={int(r['n']):,})" if r['n'] > 0 else "0.00% (n=0)",
            axis=1
        )
        display_df["Share of All Exits"] = display_df["share_of_all_exits"].apply(lambda v: f"{v:.2%}")
        
        show_cols = ["segment", "n", "exited", "retained", "Churn Rate (n)", "Share of All Exits"]
        
        # High-visibility badge flag for small sample
        if flag_small_sample:
            display_df["Reliability Tag"] = display_df["n"].apply(
                lambda n: "⚠️ Small Sample (n < 300)" if n < 300 else "✅ Adequate (n ≥ 300)"
            )
            show_cols.append("Reliability Tag")
            
        st.dataframe(
            display_df[show_cols].rename(columns={
                "segment": "Segment Cohort",
                "n": "Customers (n)",
                "exited": "Exited",
                "retained": "Retained"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        if flag_small_sample:
            small_cohorts = display_df[display_df["n"] < 300]
            if len(small_cohorts) > 0:
                examples_str = ", ".join([f"Tier {r['segment']} (n={int(r['n']):,})" for _, r in small_cohorts.iterrows()])
                max_small_rate = float(small_cohorts["churn_rate"].max())
                render_callout(
                    title="SMALL-SAMPLE DENOMINATOR WARNING (n < 300):",
                    text=f"""Product cohorts with fewer than 300 customer records (e.g., {examples_str}) 
                    produce extreme observed churn rates (up to {max_small_rate:.0%}). These denominators are statistically too small for reliable portfolio-wide generalization. 
                    Always interpret these observed percentages in the context of their exact sample count (n).""",
                    callout_type="warning",
                    icon="⚠️"
                )
                
    with col_chart:
        chart_data = table_df.copy()
        chart_data["rate_pct"] = chart_data["churn_rate"] * 100
        chart_data["display_label"] = chart_data.apply(
            lambda r: f"<b>{r['rate_pct']:.1f}%</b><br>(n={int(r['n']):,})",
            axis=1
        )
        
        colors = []
        for idx, row in chart_data.iterrows():
            if flag_small_sample and row["n"] < 300:
                colors.append(COLOR_WARNING)  # Amber warning for small sample
            elif row["rate_pct"] > (filtered_churn * 100):
                colors.append(COLOR_EXITED)   # Coral/Red for above cohort baseline
            else:
                colors.append(COLOR_RETAINED) # Teal for below cohort baseline
                
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data["segment"],
            y=chart_data["rate_pct"],
            marker_color=colors,
            marker_line=dict(color="rgba(255, 255, 255, 0.12)", width=1),
            text=chart_data["display_label"],
            textposition="outside",
            hovertemplate="<b>Cohort: %{x}</b><br>Observed Churn: %{y:.2f}%<br>Sample Size: %{text}<extra></extra>"
        ))
        
        # Cohort baseline reference line
        fig.add_hline(
            y=filtered_churn * 100,
            line_dash="dash",
            line_color=COLOR_NEUTRAL,
            annotation_text=f"Cohort Avg: {filtered_churn:.2%}",
            annotation_position="top right",
            annotation_font=dict(color=COLOR_NEUTRAL, size=10, family="Inter")
        )
        
        max_y = max(chart_data["rate_pct"].max() * 1.25, 25)
        style_plotly_figure(
            fig,
            height=290,
            yaxis_title="Observed Churn Rate (%)",
            show_legend=False
        )
        fig.update_yaxes(range=[0, min(max_y, 115)])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)


# 1. Geography Section
geo_table = summarize_geography(filtered_df, total_exits=portfolio_total_exits)
render_segment_section(
    title="1. Geography Segmentation",
    subtitle="Observed churn rate and exit volume distribution across France, Germany, and Spain",
    table_df=geo_table,
    section_icon="🌍"
)

# 2. Age Bands Section
age_table = summarize_age_bands(filtered_df, total_exits=portfolio_total_exits)
render_segment_section(
    title="2. Demographic Age Band Segmentation",
    subtitle="Observed churn distribution across standard age cohorts (<30, 30–45, 46–59, 60+)",
    table_df=age_table,
    section_icon="👥"
)

# 3. Product Count Section (with Small Sample Flagging)
prod_table = summarize_num_of_products(filtered_df, total_exits=portfolio_total_exits)
render_segment_section(
    title="3. Product Depth Segmentation (NumOfProducts)",
    subtitle="Analysis of account churn across product tiers, featuring automated small-sample (n < 300) reliability tagging",
    table_df=prod_table,
    flag_small_sample=True,
    section_icon="📦"
)

# 4. Engagement / Activity Section
active_table = summarize_is_active_member(filtered_df, total_exits=portfolio_total_exits)
render_segment_section(
    title="4. Engagement Status Segmentation (IsActiveMember)",
    subtitle="Risk differential between digitally active (1) versus inactive (0) customer accounts",
    table_df=active_table,
    section_icon="⚡"
)
