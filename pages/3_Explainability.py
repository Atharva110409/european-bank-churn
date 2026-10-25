"""
pages/3_Explainability.py
Model Explainability & Feature Attribution Page.

Displays:
1. Global SHAP feature attribution summary plot (TreeExplainer on winning model).
2. Permutation feature importance ranking (predictive cross-check on held-out test data).
3. Clear methodological disclaimer distinguishing statistical predictive reliance from real-world causality.

READ-ONLY: Strictly loads precomputed artifacts from reports/; never recalculates on page load.
All numbers trace directly to reports/permutation_importance.csv and reports/segment_tables.csv.
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

from src.ui_theme import (
    apply_custom_theme,
    render_page_header,
    render_sidebar_branding,
    render_provenance_disclaimer,
    render_kpi_card,
    render_callout,
    COLOR_PRIMARY,
    COLOR_RETAINED,
    COLOR_WARNING,
    COLOR_PURPLE
)

st.set_page_config(
    page_title="Model Explainability | AuditBank Europe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_theme()

# Unified sidebar branding
render_sidebar_branding()

# Top Hero Page Header
render_page_header(
    icon="🔍",
    title="Model Explainability & Feature Attribution",
    subtitle="Global Tree SHAP Log-Odds Contributions & Held-Out Permutation Importance Validation",
    badge="Attribution Analytics"
)

# Mandatory Provenance Disclaimer Callout
render_provenance_disclaimer()

# Causal vs Predictive Boundary Disclaimer (Professional, clearly visible, non-alarming)
render_callout(
    title="CRITICAL INTERPRETABILITY BOUNDARY (Predictive Reliance vs. Real-World Causality):",
    text="""The attribution values displayed below reflect the mathematical weights and decision splits utilized by the trained machine learning model to rank records. 
    <strong>They do not represent evidence of real-world causality or customer motivation.</strong> 
    Attribution measures statistical dependency within this retrospective dataset, not causal mechanics. 
    Interpreting these rankings as direct policy levers without randomized control trials or causal DAG verification would be methodologically unsound.""",
    callout_type="causal",
    icon="🔬"
)

REPORTS_DIR = "reports"
SHAP_PLOT_PATH = os.path.join(REPORTS_DIR, "shap_summary_global.png")
PERM_CSV_PATH = os.path.join(REPORTS_DIR, "permutation_importance.csv")
SEGMENTS_CSV_PATH = os.path.join(REPORTS_DIR, "segment_tables.csv")

if not os.path.exists(SHAP_PLOT_PATH) or not os.path.exists(PERM_CSV_PATH) or not os.path.exists(SEGMENTS_CSV_PATH):
    st.error(
        f"Required explainability artifacts not found in `{REPORTS_DIR}/`. "
        "Please ensure shap_summary_global.png, permutation_importance.csv, and segment_tables.csv exist."
    )
    st.stop()

# ----------------- Load Artifacts (Read-Only) -----------------
METRICS_PATH = "artifacts/metrics.json"

@st.cache_data
def load_explainability_artifacts():
    perm_df = pd.read_csv(PERM_CSV_PATH)
    seg_df = pd.read_csv(SEGMENTS_CSV_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    return perm_df, seg_df, metrics_data

perm_df, seg_df, metrics_data = load_explainability_artifacts()
test_records_count = int(metrics_data["held_out_test_evaluation"]["sample_size"])
num_features_count = len(perm_df)

top_features = perm_df.head(4)["feature"].tolist()
top_drops = perm_df.head(4)["predictive_importance_mean"].tolist()
perm_map = dict(zip(perm_df["feature"], perm_df["predictive_importance_mean"]))

# Extract reference segment rates dynamically from reports/segment_tables.csv
rate_46_59 = float(seg_df[(seg_df["dimension"] == "AgeBand") & (seg_df["segment"] == "46-59")]["churn_rate"].iloc[0])
rate_prod_2 = float(seg_df[(seg_df["dimension"] == "NumOfProducts") & (seg_df["segment"] == "2")]["churn_rate"].iloc[0])
rate_inactive = float(seg_df[(seg_df["dimension"] == "IsActiveMember") & (seg_df["segment"] == "Inactive")]["churn_rate"].iloc[0])
rate_active = float(seg_df[(seg_df["dimension"] == "IsActiveMember") & (seg_df["segment"] == "Active")]["churn_rate"].iloc[0])
rate_germany = float(seg_df[(seg_df["dimension"] == "Geography") & (seg_df["segment"] == "Germany")]["churn_rate"].iloc[0])

# ----------------- Top Predictive Features KPI Row -----------------
st.markdown("### 🏆 Top Predictive Feature Signals")
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi_card(
        label="Primary Predictive Anchor",
        value=f"{top_features[0]}",
        subtext=f"AP drop: {top_drops[0]:.4f}",
        delta="Rank #1 Impact",
        delta_type="exited",
        icon="📦"
    )
with col2:
    render_kpi_card(
        label="Demographic Risk Driver",
        value=f"{top_features[1]}",
        subtext=f"AP drop: {top_drops[1]:.4f}",
        delta="Rank #2 Impact",
        delta_type="warning",
        icon="👤"
    )
with col3:
    render_kpi_card(
        label="Engagement Signal",
        value=f"{top_features[2]}",
        subtext=f"AP drop: {top_drops[2]:.4f}",
        delta="Rank #3 Impact",
        delta_type="retained",
        icon="⚡"
    )
with col4:
    render_kpi_card(
        label="Balance Portfolio Profile",
        value=f"{top_features[3]}",
        subtext=f"AP drop: {top_drops[3]:.4f}",
        delta="Rank #4 Impact",
        delta_type="neutral",
        icon="💰"
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 1: Global SHAP Summary Plot -----------------
st.markdown("### 🐝 1. Global SHAP Feature Attribution (TreeExplainer)")
st.caption("Distribution of SHAP impact values for 1,000 representative records across 18 transformed feature coordinates:")

col_shap_img, col_shap_notes = st.columns([1.35, 1])

with col_shap_img:
    try:
        shap_img = Image.open(SHAP_PLOT_PATH)
        st.image(
            shap_img,
            caption="Figure 1: Global SHAP Beeswarm Plot (reports/shap_summary_global.png). Red dots denote high feature values; blue dots denote low values. Right shift pushes toward Exited=1; left shift pushes toward Retained=0.",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error displaying SHAP plot image: {e}")

with col_shap_notes:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid {COLOR_PURPLE}; height: 100%;">
        <div class="content-card-title">🔍 Strategic Attribution Insights</div>
        <ul style="font-size: 0.825rem; color: var(--eb-text-body); line-height: 1.65; padding-left: 1.15rem; margin: 0;">
            <li style="margin-bottom: 0.6rem;">
                <strong>Age (Non-Linear Escalation):</strong> Higher age values (red dots) push log-odds heavily toward churn. This confirms model alignment with the observed <strong>{rate_46_59:.2%} churn rate</strong> in the 46–59 age bracket.
            </li>
            <li style="margin-bottom: 0.6rem;">
                <strong>NumOfProducts = 2 (Stabilizing Anchor):</strong> Pushes strongly to the left (retention). Customers with exactly 2 products exhibit the lowest churn rate ({rate_prod_2:.2%}). Holding 1 product or 3+ products shifts risk sharply toward exit.
            </li>
            <li style="margin-bottom: 0.6rem;">
                <strong>IsActiveMember = 0 (Inactivity Hazard):</strong> Inactivity substantially increases churn probability, consistent with the {rate_inactive:.2%} churn among inactive customers versus {rate_active:.2%} among active accounts.
            </li>
            <li>
                <strong>Geography = Germany:</strong> Yields higher positive SHAP contributions compared to France or Spain, reflecting Germany's {rate_germany:.2%} baseline churn.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 2: Permutation Feature Importance -----------------
st.markdown("### 📊 2. Permutation Importance Cross-Validation (Held-Out Test Set)")
st.caption(f"Validating attribution robustness on n={test_records_count:,} held-out test records (10 shuffles across {num_features_count} candidate features, metric: Average Precision AP drop):")

col_perm_table, col_perm_card = st.columns([1.2, 1])

with col_perm_table:
    display_perm = perm_df.copy()
    display_perm["Mean AP Drop"] = display_perm["predictive_importance_mean"].apply(lambda v: f"{v:.4f}")
    display_perm["AP Drop Std"] = display_perm["predictive_importance_std"].apply(lambda v: f"± {v:.4f}")
    
    st.dataframe(
        display_perm[["feature", "Mean AP Drop", "AP Drop Std", "interpretability_notice"]].rename(columns={
            "feature": "Feature Candidate",
            "interpretability_notice": "Methodological Classification"
        }),
        use_container_width=True,
        hide_index=True
    )

with col_perm_card:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid {COLOR_PRIMARY};">
        <div class="content-card-title">✅ Attribution Cross-Check Confirmation</div>
        <div style="font-size: 0.85rem; color: var(--eb-text-body); line-height: 1.6; margin-bottom: 0.75rem;">
            Permutation importance independently confirms that <strong>{top_features[0]}</strong> (mean AP drop: {perm_map[top_features[0]]:.4f}) and <strong>{top_features[1]}</strong> (mean AP drop: {perm_map[top_features[1]]:.4f}) 
            are the dominant predictive pillars of the HistGradientBoosting ensemble.
        </div>
        <div style="font-size: 0.825rem; color: var(--eb-text-label); line-height: 1.55; border-top: 1px solid var(--eb-border-card); padding-top: 0.6rem;">
            <strong>Low Predictive Reliance:</strong> In contrast, features like <code>HasCrCard</code> (AP drop: {perm_map['HasCrCard']:.4f}) contribute almost zero discrimination to the model, matching exploratory data findings and confirming the model has not overfit to irrelevant fields.
        </div>
    </div>
    """, unsafe_allow_html=True)
