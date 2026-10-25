"""
explainability.py
Post-hoc model interpretability and feature attribution.

Loads the saved model pipeline and training data to compute:
1. Global SHAP values via TreeExplainer for the winning HistGradientBoostingClassifier.
2. Global SHAP summary beeswarm/bar plot saved to reports/shap_summary_global.png.
3. Permutation importance cross-check table saved to reports/permutation_importance.csv.

CRITICAL METHODOLOGICAL POLICY:
All metrics generated in this module reflect strictly predictive importance
(the statistical signals the model utilizes for risk ranking), NOT causal importance
(they do not demonstrate what causes customer churn).
"""

import os
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data_validation import load_and_validate_data, get_features_and_target

REPORTS_DIR = "reports"
DEFAULT_MODEL_PATH = "artifacts/model.joblib"
DEFAULT_DATA_PATH = "data/raw/European_Bank.csv"


def clean_feature_name(name: str) -> str:
    """
    Cleans ColumnTransformer output feature names for clear visualization.
    e.g. 'num__Age' -> 'Age', 'cat__Geography_Germany' -> 'Geography: Germany'
    """
    if name.startswith("num__"):
        return name.replace("num__", "")
    elif name.startswith("cat__"):
        clean = name.replace("cat__", "")
        if "_" in clean:
            parts = clean.split("_", 1)
            return f"{parts[0]} = {parts[1]}"
        return clean
    return name


def compute_explainability(
    model_path: str = DEFAULT_MODEL_PATH,
    csv_path: str = DEFAULT_DATA_PATH,
    reports_dir: str = REPORTS_DIR,
    sample_size: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes global SHAP values and permutation importance for the saved model.
    Saves:
      - reports/shap_summary_global.png
      - reports/permutation_importance.csv
    Returns a dictionary of explainability outputs.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Load data and extract features
    df, _ = load_and_validate_data(csv_path)
    X, y = get_features_and_target(df)
    
    # Split using the exact project protocol (80/20 train/test, stratify=y, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    # Sample training records for efficient SHAP explanation
    np.random.seed(random_state)
    if len(X_train) > sample_size:
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]
    else:
        X_sample = X_train
        y_sample = y_train

    # 2. Load fitted pipeline
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    
    # Transform sample features
    X_trans = preprocessor.transform(X_sample)
    raw_feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [clean_feature_name(n) for n in raw_feature_names]
    
    # 3. Compute TreeExplainer SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_trans)
    
    # If binary classification returns list or 3D array, isolate positive class (churn)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_values_pos = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_values_pos = shap_vals[:, :, 1]
    else:
        shap_values_pos = shap_vals

    # 4. Generate and save SHAP summary plot
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(
        shap_values_pos,
        features=X_trans,
        feature_names=cleaned_feature_names,
        show=False,
        plot_type="dot",
        max_display=12
    )
    plt.title(
        "Global SHAP Value Attribution (Predictive Importance, Not Causal)\n"
        "Model: HistGradientBoostingClassifier | Positive Class: Exited=1",
        fontsize=11,
        pad=15
    )
    plt.xlabel("SHAP value (impact on model log-odds output)")
    plt.tight_layout()
    
    shap_plot_path = os.path.join(reports_dir, "shap_summary_global.png")
    plt.savefig(shap_plot_path, bbox_inches="tight")
    plt.close()
    
    # Mean absolute SHAP importance table
    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": cleaned_feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    # 5. Compute Permutation Importance on held-out test data (predictive cross-check)
    perm_result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="average_precision"
    )
    
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "predictive_importance_mean": perm_result.importances_mean,
        "predictive_importance_std": perm_result.importances_std,
        "interpretability_notice": "Predictive importance, not causal importance"
    }).sort_values(by="predictive_importance_mean", ascending=False).reset_index(drop=True)
    
    perm_csv_path = os.path.join(reports_dir, "permutation_importance.csv")
    perm_df.to_csv(perm_csv_path, index=False)
    
    return {
        "shap_plot_path": shap_plot_path,
        "perm_csv_path": perm_csv_path,
        "shap_importance_df": shap_importance_df,
        "permutation_importance_df": perm_df,
        "shap_values_shape": shap_values_pos.shape,
        "feature_count": len(cleaned_feature_names)
    }


if __name__ == "__main__":
    print("Running explainability analysis...")
    res = compute_explainability()
    print("Explainability analysis complete:")
    print(f" - SHAP plot saved to: {res['shap_plot_path']}")
    print(f" - Permutation importance saved to: {res['perm_csv_path']}")
    print("\nTop 5 Features by Predictive Permutation Importance (AP Drop):")
    print(res["permutation_importance_df"].head(5).to_string(index=False))
