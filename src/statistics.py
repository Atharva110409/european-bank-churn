"""
statistics.py
Rigorous hypothesis testing, multiple comparison correction, and bootstrap inference.

Provides:
1. Chi-square tests of independence with Cramér's V effect sizes across all 4 segment dimensions.
2. Benjamini-Hochberg False Discovery Rate (FDR) correction (Benjamini & Hochberg 1995) at q=0.05.
3. 1,000-resample non-parametric bootstrap 95% confidence intervals (fixed seed=42) for:
   - Overall churn rate
   - Age 46–59 cohort churn rate
   - Held-out test Average Precision (AP)
   - Held-out test ROC-AUC
4. Exports:
   - reports/segment_significance_tests.csv
   - reports/bootstrap_cis.json
"""

import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_validation import load_and_validate_data
from src.segmentation import assign_age_band

REPORTS_DIR = "reports"
DEFAULT_DATA_PATH = "data/raw/European_Bank.csv"
DEFAULT_PREDS_PATH = "artifacts/test_predictions.csv"
RANDOM_SEED = 42
N_BOOTSTRAP = 1000


def benjamini_hochberg_correction(raw_p_values: List[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Applies Benjamini-Hochberg (1995) FDR control procedure.
    Returns: (adjusted_p_values, is_significant_flags)
    Ensures monotonicity: adjusted_p[i] >= raw_p[i].
    """
    m = len(raw_p_values)
    p_array = np.array(raw_p_values, dtype=float)
    
    # Sort p-values while preserving original indices
    sort_order = np.argsort(p_array)
    sorted_p = p_array[sort_order]
    
    # Calculate adjusted p: p_(i) * m / i
    ranks = np.arange(1, m + 1)
    adj_p_sorted = sorted_p * (m / ranks)
    
    # Enforce monotonicity from right to left
    adj_p_sorted = np.minimum.accumulate(adj_p_sorted[::-1])[::-1]
    adj_p_sorted = np.minimum(adj_p_sorted, 1.0)
    # Ensure adjusted p is never strictly less than raw p
    adj_p_sorted = np.maximum(adj_p_sorted, sorted_p)
    
    # Restore original order
    inv_order = np.argsort(sort_order)
    adjusted_p = adj_p_sorted[inv_order].tolist()
    significant = [bool(p_adj < q) for p_adj in adjusted_p]
    
    return adjusted_p, significant


def run_segment_significance_tests(
    df: pd.DataFrame,
    reports_dir: str = REPORTS_DIR,
    q_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Computes Chi-square tests of independence, Cramér's V, and BH FDR correction
    across Geography, AgeBand, NumOfProducts, and IsActiveMember.
    Saves results to reports/segment_significance_tests.csv.
    """
    os.makedirs(reports_dir, exist_ok=True)
    df_work = df.copy()
    if "AgeBand" not in df_work.columns:
        df_work["AgeBand"] = assign_age_band(df_work["Age"])
        
    dimensions = ["Geography", "AgeBand", "NumOfProducts", "IsActiveMember"]
    raw_results = []
    raw_p_values = []
    
    for dim in dimensions:
        contingency = pd.crosstab(df_work[dim], df_work["Exited"])
        chi2, p_val, dof, _ = chi2_contingency(contingency)
        n = float(contingency.values.sum())
        min_dim = min(contingency.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
        
        raw_results.append({
            "dimension": dim,
            "chi2_statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "raw_p_value": float(p_val),
            "cramers_v": float(cramers_v)
        })
        raw_p_values.append(float(p_val))
        
    adj_p_values, sig_flags = benjamini_hochberg_correction(raw_p_values, q=q_threshold)
    
    for i, r in enumerate(raw_results):
        r["fdr_adjusted_p_value"] = adj_p_values[i]
        r["significant_fdr_05"] = sig_flags[i]
        
    res_df = pd.DataFrame(raw_results)
    out_csv = os.path.join(reports_dir, "segment_significance_tests.csv")
    res_df.to_csv(out_csv, index=False)
    return res_df


def compute_bootstrap_confidence_intervals(
    df: pd.DataFrame,
    preds_df: pd.DataFrame,
    reports_dir: str = REPORTS_DIR,
    n_resamples: int = N_BOOTSTRAP,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Computes 1,000-resample non-parametric percentile bootstrap 95% CIs for:
    - Overall churn rate
    - Age 46–59 churn rate
    - Held-out test Average Precision (AP)
    - Held-out test ROC-AUC
    Saves results to reports/bootstrap_cis.json.
    """
    os.makedirs(reports_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)
    
    df_work = df.copy()
    if "AgeBand" not in df_work.columns:
        df_work["AgeBand"] = assign_age_band(df_work["Age"])
        
    # 1. Overall Churn Rate
    overall_exited = df_work["Exited"].values
    n_overall = len(overall_exited)
    boot_overall = [
        float(np.mean(rng.choice(overall_exited, size=n_overall, replace=True)))
        for _ in range(n_resamples)
    ]
    ci_overall = [float(x) for x in np.percentile(boot_overall, [2.5, 97.5])]
    point_overall = float(np.mean(overall_exited))
    
    # 2. Age 46-59 Cohort Churn Rate
    age_46_59_exited = df_work[df_work["AgeBand"] == "46-59"]["Exited"].values
    n_46_59 = len(age_46_59_exited)
    boot_46_59 = [
        float(np.mean(rng.choice(age_46_59_exited, size=n_46_59, replace=True)))
        for _ in range(n_resamples)
    ]
    ci_46_59 = [float(x) for x in np.percentile(boot_46_59, [2.5, 97.5])]
    point_46_59 = float(np.mean(age_46_59_exited))
    
    # 3 & 4. Test Set AP and ROC-AUC
    y_true = preds_df["y_true"].values
    y_prob = preds_df["predicted_probability"].values
    n_test = len(preds_df)
    
    boot_ap = []
    boot_roc = []
    for _ in range(n_resamples):
        indices = rng.choice(n_test, size=n_test, replace=True)
        y_t_boot = y_true[indices]
        y_p_boot = y_prob[indices]
        # Guard against zero-positive resample
        if len(np.unique(y_t_boot)) < 2:
            continue
        boot_ap.append(float(average_precision_score(y_t_boot, y_p_boot)))
        boot_roc.append(float(roc_auc_score(y_t_boot, y_p_boot)))
        
    ci_ap = [float(x) for x in np.percentile(boot_ap, [2.5, 97.5])]
    point_ap = float(average_precision_score(y_true, y_prob))
    
    ci_roc = [float(x) for x in np.percentile(boot_roc, [2.5, 97.5])]
    point_roc = float(roc_auc_score(y_true, y_prob))
    
    cis_payload = {
        "metadata": {
            "resamples": n_resamples,
            "random_seed": random_state,
            "confidence_level": 0.95,
            "ci_method": "percentile"
        },
        "overall_churn_rate": {
            "point_estimate": point_overall,
            "ci_lower_95": ci_overall[0],
            "ci_upper_95": ci_overall[1]
        },
        "age_46_59_churn_rate": {
            "point_estimate": point_46_59,
            "ci_lower_95": ci_46_59[0],
            "ci_upper_95": ci_46_59[1]
        },
        "test_average_precision_ap": {
            "point_estimate": point_ap,
            "ci_lower_95": ci_ap[0],
            "ci_upper_95": ci_ap[1]
        },
        "test_roc_auc": {
            "point_estimate": point_roc,
            "ci_lower_95": ci_roc[0],
            "ci_upper_95": ci_roc[1]
        }
    }
    
    out_json = os.path.join(reports_dir, "bootstrap_cis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cis_payload, f, indent=2)
        
    return cis_payload


def run_full_statistical_audit(
    data_path: str = DEFAULT_DATA_PATH,
    preds_path: str = DEFAULT_PREDS_PATH,
    reports_dir: str = REPORTS_DIR
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs full statistical audit and generates all testing artifacts.
    """
    df, _ = load_and_validate_data(data_path)
    preds_df = pd.read_csv(preds_path)
    
    sig_df = run_segment_significance_tests(df, reports_dir=reports_dir)
    cis = compute_bootstrap_confidence_intervals(df, preds_df, reports_dir=reports_dir)
    
    return sig_df, cis


if __name__ == "__main__":
    print("Executing statistical audit and bootstrap estimation...")
    sig_df, cis = run_full_statistical_audit()
    print("\n--- Chi-Square Tests of Independence & FDR Correction ---")
    print(sig_df.to_string(index=False))
    print("\n--- 1,000-Resample Bootstrap 95% Confidence Intervals ---")
    print(json.dumps(cis, indent=2))
