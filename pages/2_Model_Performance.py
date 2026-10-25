"""
pages/2_Model_Performance.py
Model Evaluation & Selection Protocol Page.

Reads saved artifacts only (artifacts/metrics.json and artifacts/test_predictions.csv).
Never fits or refits a model on page load.
Displays:
- Cross-validation comparison table with winning model visually highlighted.
- Clear note distinguishing CV training metrics from held-out test evaluation.
- Final test metrics for the selected model vs Dummy baseline.
- Clean side-by-side card layout comparing 0.50 default vs 0.27 alternative thresholds.
- Confusion matrix heatmap with diagnostic rates.
- ROC Curve and Precision-Recall Curve computed from full probability predictions.
All numbers trace directly to artifacts/metrics.json and artifacts/test_predictions.csv.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

from src.ui_theme import (
    apply_custom_theme,
    render_page_header,
    render_sidebar_branding,
    render_provenance_disclaimer,
    render_kpi_card,
    render_callout,
    style_plotly_figure,
    COLOR_PRIMARY,
    COLOR_PRIMARY_LIGHT,
    COLOR_RETAINED,
    COLOR_EXITED,
    COLOR_WARNING,
    COLOR_NEUTRAL
)

st.set_page_config(
    page_title="Model Performance | AuditBank Europe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling
apply_custom_theme()

# Unified sidebar branding
render_sidebar_branding()

# Top Hero Page Header
render_page_header(
    icon="📈",
    title="Predictive Model Performance & Evaluation",
    subtitle="Cross-Validation Model Selection Protocol & Held-Out Test Generalization Audit",
    badge="Model Intelligence"
)

# Mandatory Provenance Disclaimer Callout
render_provenance_disclaimer()

# ----------------- Load Artifacts (Read-Only) -----------------
METRICS_PATH = "artifacts/metrics.json"
PREDICTIONS_PATH = "artifacts/test_predictions.csv"

if not os.path.exists(METRICS_PATH) or not os.path.exists(PREDICTIONS_PATH):
    st.error("Evaluation artifacts not found. Please ensure artifacts/metrics.json and test_predictions.csv exist.")
    st.stop()

@st.cache_data
def load_artifacts():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    preds_df = pd.read_csv(PREDICTIONS_PATH)
    return metrics_data, preds_df

metrics, test_preds = load_artifacts()

selected_model = metrics["model_selection"]["selected_model"]
selection_rationale = metrics["model_selection"]["rationale"]
cv_comp = metrics["cv_comparison"]
test_eval = metrics["held_out_test_evaluation"]
test_sample_size = int(test_eval["sample_size"])
train_records = 10000 - test_sample_size
train_pct = train_records / 10000
test_pct = test_sample_size / 10000

m05 = test_eval["metrics_at_default_threshold_0_5"]
malt = test_eval["cv_selected_alternative_threshold"]
dummy_ref = test_eval["dummy_baseline_reference"]

cm5 = m05["confusion_matrix"]
cm_alt = malt["confusion_matrix"]
total_test_exited = cm5["TP"] + cm5["FN"]
total_test_retained = cm5["TN"] + cm5["FP"]

# Methodological Discipline Notice
render_callout(
    title="METHODOLOGICAL DISCIPLINE & GENERALIZATION INTEGRITY:",
    text=f"""Candidate model selection was governed <strong>exclusively by 5-fold cross-validated Average Precision (AP) on the {train_pct:.0%} training partition (n={train_records:,})</strong>. 
    The {test_pct:.0%} held-out test split (n={test_sample_size:,}) was locked away and evaluated <strong>strictly once</strong> after all model specifications were finalized. 
    Held-out test performance was <em>never</em> used to tune hyper-parameters or influence model selection.""",
    callout_type="methodology",
    icon="🛡️"
)

# ----------------- SECTION 1: 5-Fold Cross-Validation Comparison -----------------
st.markdown("### 🏆 1. Cross-Validation Model Selection Ladder")
st.caption(f"5-fold `StratifiedKFold(shuffle=True, random_state=42)` across {train_records:,} training records:")

cv_rows = []
for model_key, scores in cv_comp.items():
    if model_key == selected_model:
        status_label = "🏆 Winner (Selected)"
    elif model_key == "Dummy":
        status_label = "Baseline Reference"
    else:
        status_label = "Candidate"
        
    cv_rows.append({
        "Model Candidate": model_key,
        "Status": status_label,
        "Mean AP ± Std": f"{scores['mean_AP']:.4f} ± {scores['std_AP']:.4f}",
        "Mean ROC-AUC ± Std": f"{scores['mean_ROC_AUC']:.4f} ± {scores['std_ROC_AUC']:.4f}",
        "raw_ap": scores["mean_AP"],
        "raw_roc": scores["mean_ROC_AUC"]
    })

cv_df = pd.DataFrame(cv_rows)

col_cv_table, col_cv_rationale = st.columns([1.2, 1])

with col_cv_table:
    # Stylized CV table
    st.dataframe(
        cv_df[["Model Candidate", "Status", "Mean AP ± Std", "Mean ROC-AUC ± Std"]],
        use_container_width=True,
        hide_index=True
    )

with col_cv_rationale:
    st.markdown(f"""
    <div class="content-card" style="border-left: 4px solid {COLOR_RETAINED};">
        <div class="content-card-title">🎯 Model Selection Rationale</div>
        <div style="font-size: 0.875rem; color: var(--eb-text-title); line-height: 1.5; margin-bottom: 0.6rem;">
            {selection_rationale}
        </div>
        <div style="font-size: 0.775rem; color: var(--eb-text-label); line-height: 1.45; border-top: 1px solid var(--eb-border-card); padding-top: 0.5rem;">
            <strong>Decision Rule:</strong> If candidates scored within 0.01 AP of each other, Logistic Regression was designated for interpretability. 
            HistGradientBoosting achieved a decisive margin of <strong>+{(cv_comp['HistGradientBoosting']['mean_AP'] - cv_comp['LogisticRegression']['mean_AP']):.4f} AP</strong>, justifying the ensemble choice.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 2: Final Held-Out Test Evaluation -----------------
st.markdown(f"### 🎯 2. Held-Out Test Generalization: {selected_model}")
st.caption(f"Evaluated once on {test_sample_size:,} held-out customer records (Exited: {total_test_exited:,}, Retained: {total_test_retained:,})")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_kpi_card(
        label="Test Average Precision (AP)",
        value=f"{test_eval['AP']:.4f}",
        subtext=f"Dummy baseline: {dummy_ref['AP']:.4f}",
        delta=f"+{(test_eval['AP'] - dummy_ref['AP']):.4f} lift",
        delta_type="retained",
        icon="🎯"
    )
with kpi2:
    render_kpi_card(
        label="Test ROC-AUC",
        value=f"{test_eval['ROC_AUC']:.4f}",
        subtext=f"Dummy baseline: {dummy_ref['ROC_AUC']:.4f}",
        delta=f"+{(test_eval['ROC_AUC'] - dummy_ref['ROC_AUC']):.4f} lift",
        delta_type="retained",
        icon="📈"
    )
with kpi3:
    render_kpi_card(
        label=f"F1 Score (@ {m05['threshold']:.2f} Default)",
        value=f"{m05['f1']:.4f}",
        subtext=f"CV-alt F1 (@ {malt['threshold']:.2f}): {malt['f1']:.4f}",
        delta="Default Operating Point",
        delta_type="neutral",
        icon="⚖️"
    )
with kpi4:
    render_kpi_card(
        label=f"Precision / Recall (@ {m05['threshold']:.2f})",
        value=f"{m05['precision']:.1%} / {m05['recall']:.1%}",
        subtext=f"CV-alt: {malt['precision']:.1%} / {malt['recall']:.1%}",
        delta="Precision Priority",
        delta_type="warning",
        icon="🔍"
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 3: Side-by-Side Threshold Comparison -----------------
st.markdown(f"### ⚖️ 3. Operational Threshold Trade-Off Analysis: {m05['threshold']:.2f} vs {malt['threshold']:.2f}")

render_callout(
    title="OPERATIONAL DECISION BOUNDARY COMPARISON:",
    text=f"""The classification decision threshold directly governs the balance between false alarms and missed exits. 
    Below, we present a side-by-side comparison between the standard conservative threshold ({m05['threshold']:.2f}) and the CV-optimized F1 threshold ({malt['threshold']:.2f}).""",
    callout_type="tradeoff",
    icon="⚖️"
)

col_th1, col_th2 = st.columns(2)

with col_th1:
    st.markdown(f"""
    <div class="content-card" style="border-top: 4px solid {COLOR_PRIMARY}; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-family: var(--eb-font-heading); font-size: 1.15rem; font-weight: 800; color: var(--eb-text-title);">
                Default Threshold: {m05['threshold']:.2f}
            </div>
            <span class="badge badge-info">Reported Default</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--eb-text-label); margin-bottom: 1rem;">
            <strong>Operational Stance:</strong> High Precision / Conservative Outreach. Minimizes unnecessary marketing contact to loyal customers.
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="background: var(--eb-bg-card-subtle); padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid var(--eb-border-card);">
                <div style="font-size: 0.7rem; color: var(--eb-text-label); text-transform: uppercase;">Precision</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {COLOR_RETAINED};">{m05['precision']:.1%}</div>
                <div style="font-size: 0.7rem; color: var(--eb-text-subtle);">Nearly 3 of 4 flags correct</div>
            </div>
            <div style="background: var(--eb-bg-card-subtle); padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid var(--eb-border-card);">
                <div style="font-size: 0.7rem; color: var(--eb-text-label); text-transform: uppercase;">Recall (Detection)</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {COLOR_WARNING};">{m05['recall']:.1%}</div>
                <div style="font-size: 0.7rem; color: var(--eb-text-subtle);">Catches {cm5['TP']:,} of {total_test_exited:,} churners</div>
            </div>
        </div>
        <div style="font-size: 0.825rem; color: var(--eb-text-body); line-height: 1.6;">
            <div>• <strong>Exits Caught (TP):</strong> {cm5['TP']:,} customers</div>
            <div>• <strong>Exits Missed (FN):</strong> {cm5['FN']:,} customers</div>
            <div>• <strong>False Alarms (FP):</strong> {cm5['FP']:,} customers (Low friction)</div>
            <div>• <strong>F1 Score:</strong> {m05['f1']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_th2:
    st.markdown(f"""
    <div class="content-card" style="border-top: 4px solid {COLOR_WARNING}; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-family: var(--eb-font-heading); font-size: 1.15rem; font-weight: 800; color: var(--eb-text-title);">
                CV-Alternative Threshold: {malt['threshold']:.2f}
            </div>
            <span class="badge badge-warning">Max CV F1</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--eb-text-label); margin-bottom: 1rem;">
            <strong>Operational Stance:</strong> High Recall / Proactive Capture. Maximizes exit detection when churn cost greatly exceeds contact cost.
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="background: var(--eb-bg-card-subtle); padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid var(--eb-border-card);">
                <div style="font-size: 0.7rem; color: var(--eb-text-label); text-transform: uppercase;">Precision</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {COLOR_WARNING};">{malt['precision']:.1%}</div>
                <div style="font-size: 0.7rem; color: var(--eb-text-subtle);">Moderate false positive rate</div>
            </div>
            <div style="background: var(--eb-bg-card-subtle); padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid var(--eb-border-card);">
                <div style="font-size: 0.7rem; color: var(--eb-text-label); text-transform: uppercase;">Recall (Detection)</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {COLOR_RETAINED};">{malt['recall']:.1%}</div>
                <div style="font-size: 0.7rem; color: var(--eb-text-subtle);">Catches {cm_alt['TP']:,} of {total_test_exited:,} churners (+{cm_alt['TP'] - cm5['TP']:,})</div>
            </div>
        </div>
        <div style="font-size: 0.825rem; color: var(--eb-text-body); line-height: 1.6;">
            <div>• <strong>Exits Caught (TP):</strong> {cm_alt['TP']:,} customers (+{((cm_alt['TP'] - cm5['TP']) / cm5['TP']):.1%} gain)</div>
            <div>• <strong>Exits Missed (FN):</strong> {cm_alt['FN']:,} customers (Fewer missed)</div>
            <div>• <strong>False Alarms (FP):</strong> {cm_alt['FP']:,} customers (+{cm_alt['FP'] - cm5['FP']:,} false alarms)</div>
            <div>• <strong>F1 Score:</strong> {malt['f1']:.4f} (CV-optimal balance)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 4: Confusion Matrix Heatmap -----------------
st.markdown(f"### 🔲 4. Held-Out Confusion Matrix Heatmap (Default Threshold {m05['threshold']:.2f})")

col_cm1, col_cm2 = st.columns([1, 1.2])

with col_cm1:
    cm_matrix = np.array([
        [cm5["TN"], cm5["FP"]],
        [cm5["FN"], cm5["TP"]]
    ])
    
    # Elegant custom heatmap
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=["Predicted Retained (0)", "Predicted Exited (1)"],
        y=["Actual Retained (0)", "Actual Exited (1)"],
        text=[
            [f"<b>True Negative (TN)</b><br>{cm5['TN']:,}<br>({cm5['TN']/total_test_retained:.1%} retained)", f"<b>False Alarm (FP)</b><br>{cm5['FP']:,}<br>({cm5['FP']/total_test_retained:.1%} flagged)"],
            [f"<b>Missed Exit (FN)</b><br>{cm5['FN']:,}<br>({cm5['FN']/total_test_exited:.1%} missed)", f"<b>Caught Churn (TP)</b><br>{cm5['TP']:,}<br>({cm5['TP']/total_test_exited:.1%} caught)"]
        ],
        texttemplate="%{text}",
        textfont={"size": 12, "family": "Inter"},
        colorscale=[
            [0.0, "rgba(99, 102, 241, 0.15)"],
            [0.3, "rgba(99, 102, 241, 0.40)"],
            [0.7, "rgba(99, 102, 241, 0.70)"],
            [1.0, "rgba(99, 102, 241, 0.95)"]
        ],
        showscale=False
    ))
    
    style_plotly_figure(fig_cm, height=290, show_legend=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm2:
    st.markdown(f"""
    <div class="content-card">
        <div class="content-card-title">🔍 Diagnostic Interpretation</div>
        <ul style="font-size: 0.85rem; color: var(--eb-text-body); line-height: 1.65; padding-left: 1.2rem; margin: 0;">
            <li><strong>High Confidence When Flagged ({m05['precision']:.1%} Precision):</strong> When an account is classified as Exited=1 at threshold {m05['threshold']:.2f}, nearly three out of four actually leave (only {cm5['FP']:,} false alarms across {total_test_retained:,} retained customers).</li>
            <li><strong>Conservative Churn Capture ({m05['recall']:.1%} Recall):</strong> The model prioritizes precision, resulting in {cm5['FN']:,} churned customers remaining unflagged as False Negatives.</li>
            <li><strong>Operational Leverage:</strong> Shifting to threshold {malt['threshold']:.2f} captures <strong>{cm_alt['TP'] - cm5['TP']:,} additional churned customers</strong> (reaching {malt['recall']:.1%} recall, {cm_alt['TP']:,} total caught), requiring customer success teams to manage {cm_alt['FP'] - cm5['FP']:,} additional outreach contacts ({cm_alt['FP']:,} total).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 5: ROC & PR Curves -----------------
st.markdown("### 📈 5. Discrimination Curves (Full Probability Predictions)")
st.caption("Plotted directly from saved held-out test predictions (y_true and predicted_probability)")

y_true = test_preds["y_true"].values
y_prob = test_preds["predicted_probability"].values

fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

col_roc, col_pr = st.columns(2)

with col_roc:
    fig_roc = go.Figure()
    
    # Model ROC line
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{selected_model} (AUC = {test_eval['ROC_AUC']:.4f})",
        line=dict(color=COLOR_PRIMARY, width=3),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
    ))
    
    # Random chance line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name=f"Random Chance (AUC = {dummy_ref['ROC_AUC']:.4f})",
        line=dict(color=COLOR_NEUTRAL, width=1.5, dash="dash"),
        hoverinfo="skip"
    ))
    
    style_plotly_figure(
        fig_roc,
        height=380,
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        show_legend=True
    )
    fig_roc.update_xaxes(range=[-0.02, 1.02])
    fig_roc.update_yaxes(range=[-0.02, 1.05])
    fig_roc.update_layout(legend=dict(x=0.38, y=0.12, bgcolor="rgba(15,23,42,0.7)"))
    st.plotly_chart(fig_roc, use_container_width=True)

with col_pr:
    fig_pr = go.Figure()
    
    # Model PR line
    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name=f"{selected_model} (AP = {test_eval['AP']:.4f})",
        line=dict(color=COLOR_RETAINED, width=3),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
    ))
    
    # Prevalence baseline
    test_prevalence = float(np.mean(y_true))
    fig_pr.add_hline(
        y=test_prevalence,
        line_dash="dash",
        line_color=COLOR_NEUTRAL,
        annotation_text=f"Baseline Prevalence: {test_prevalence:.2%}",
        annotation_position="bottom right",
        annotation_font=dict(color=COLOR_NEUTRAL, size=10, family="Inter")
    )
    
    # Operating point at default threshold 0.50
    fig_pr.add_trace(go.Scatter(
        x=[m05["recall"]],
        y=[m05["precision"]],
        mode="markers",
        name=f"Operating Point (th={m05['threshold']:.2f})",
        marker=dict(color=COLOR_EXITED, size=11, symbol="diamond", line=dict(color="white", width=1.5)),
        hovertemplate=f"Th {m05['threshold']:.2f}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
    ))
    
    # Operating point at CV alternative
    fig_pr.add_trace(go.Scatter(
        x=[malt["recall"]],
        y=[malt["precision"]],
        mode="markers",
        name=f"CV Alternative (th={malt['threshold']:.2f})",
        marker=dict(color=COLOR_WARNING, size=11, symbol="circle", line=dict(color="white", width=1.5)),
        hovertemplate=f"Th {malt['threshold']:.2f}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
    ))
    
    style_plotly_figure(
        fig_pr,
        height=380,
        title="Precision-Recall Curve (PR)",
        xaxis_title="Recall (Coverage of Exits)",
        yaxis_title="Precision (Accuracy of Flags)",
        show_legend=True
    )
    fig_pr.update_xaxes(range=[-0.02, 1.02])
    fig_pr.update_yaxes(range=[-0.02, 1.05])
    fig_pr.update_layout(legend=dict(x=0.03, y=0.18, bgcolor="rgba(15,23,42,0.7)"))
    st.plotly_chart(fig_pr, use_container_width=True)
