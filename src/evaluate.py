"""
evaluate.py
Held-out test set evaluation and artifact generation.

Follows the blueprint:
- Evaluates the selected pipeline once on the held-out test split.
- Evaluates the Dummy baseline on test as a reference only.
- Computes metrics at threshold 0.5 labeled "default threshold, not necessarily optimal".
- Separately reports metrics at the CV-selected alternative threshold (for comparison only).
- Saves:
  1) artifacts/model.joblib: identical fitted pipeline instance from training.
  2) artifacts/metrics.json: complete CV comparison, test metrics, baseline metrics, threshold data.
  3) artifacts/test_predictions.csv: anonymous y_true, predicted_probability, predicted_label only.
"""

import os
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.train import train_and_select_model

ARTIFACTS_DIR = "artifacts"


def evaluate_model_on_test(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_alt_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates a fitted pipeline on held-out test data.
    Computes AP, ROC-AUC, metrics at default threshold 0.5, and metrics at CV alternative threshold.
    """
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    
    # Ranking metrics
    ap = float(average_precision_score(y_test, test_probs))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    
    # Metrics at default threshold 0.5
    preds_05 = (test_probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_test, preds_05)
    metrics_05 = {
        "label": "default threshold, not necessarily optimal",
        "threshold": 0.5,
        "precision": float(precision_score(y_test, preds_05, zero_division=0)),
        "recall": float(recall_score(y_test, preds_05, zero_division=0)),
        "f1": float(f1_score(y_test, preds_05, zero_division=0)),
        "confusion_matrix": {
            "TN": int(cm_05[0, 0]),
            "FP": int(cm_05[0, 1]),
            "FN": int(cm_05[1, 0]),
            "TP": int(cm_05[1, 1])
        }
    }
    
    # Metrics at CV-selected alternative threshold
    preds_alt = (test_probs >= cv_alt_threshold).astype(int)
    cm_alt = confusion_matrix(y_test, preds_alt)
    metrics_alt = {
        "label": "CV-selected alternative threshold (comparison only)",
        "threshold": float(cv_alt_threshold),
        "precision": float(precision_score(y_test, preds_alt, zero_division=0)),
        "recall": float(recall_score(y_test, preds_alt, zero_division=0)),
        "f1": float(f1_score(y_test, preds_alt, zero_division=0)),
        "confusion_matrix": {
            "TN": int(cm_alt[0, 0]),
            "FP": int(cm_alt[0, 1]),
            "FN": int(cm_alt[1, 0]),
            "TP": int(cm_alt[1, 1])
        }
    }
    
    return {
        "AP": ap,
        "ROC_AUC": roc_auc,
        "metrics_at_default_threshold_0_5": metrics_05,
        "metrics_at_cv_selected_threshold": metrics_alt,
        "test_probabilities": test_probs,
        "test_predictions_05": preds_05
    }


def evaluate_dummy_baseline_on_test(
    dummy_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates the DummyClassifier(strategy="prior") on held-out test data.
    """
    test_probs = dummy_pipeline.predict_proba(X_test)[:, 1]
    preds = dummy_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    
    return {
        "AP": float(average_precision_score(y_test, test_probs)),
        "ROC_AUC": float(roc_auc_score(y_test, test_probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": {
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1])
        }
    }


def run_evaluation(
    train_results: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Runs full evaluation without test-set leakage:
    1. Obtains trained model and splits from train_and_select_model.
    2. Evaluates selected pipeline once on held-out test split.
    3. Evaluates dummy baseline once on held-out test split.
    4. Confirms model identity and persists artifacts.
    """
    if train_results is None:
        train_results = train_and_select_model()
        
    selected_pipeline = train_results["selected_pipeline"]
    dummy_pipeline = train_results["dummy_pipeline"]
    selected_name = train_results["selected_model_name"]
    cv_alt_threshold = train_results["cv_optimal_threshold"]["threshold"]
    
    X_test = train_results["X_test"]
    y_test = train_results["y_test"]
    
    # Evaluate once on held-out test
    eval_selected = evaluate_model_on_test(
        pipeline=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
        cv_alt_threshold=cv_alt_threshold
    )
    
    eval_dummy = evaluate_dummy_baseline_on_test(
        dummy_pipeline=dummy_pipeline,
        X_test=X_test,
        y_test=y_test
    )
    
    # Prepare metrics.json payload
    metrics_payload = {
        "model_selection": {
            "metric": "Average Precision (AP)",
            "protocol": (
                "5-fold StratifiedKFold on training partition only (random_state=42). "
                "The held-out test set is never used to select or tune the model. "
                "Tie-breaker rule: if candidate AP scores are within 0.01 of each other, "
                "Logistic Regression is preferred for interpretability."
            ),
            "selected_model": selected_name,
            "rationale": train_results["selection_rationale"]
        },
        "cv_comparison": {
            "Dummy": {
                "mean_AP": train_results["cv_metrics"]["Dummy"]["mean_AP"],
                "std_AP": train_results["cv_metrics"]["Dummy"]["std_AP"],
                "mean_ROC_AUC": train_results["cv_metrics"]["Dummy"]["mean_ROC_AUC"],
                "std_ROC_AUC": train_results["cv_metrics"]["Dummy"]["std_ROC_AUC"]
            },
            "LogisticRegression": {
                "mean_AP": train_results["cv_metrics"]["LogisticRegression"]["mean_AP"],
                "std_AP": train_results["cv_metrics"]["LogisticRegression"]["std_AP"],
                "mean_ROC_AUC": train_results["cv_metrics"]["LogisticRegression"]["mean_ROC_AUC"],
                "std_ROC_AUC": train_results["cv_metrics"]["LogisticRegression"]["std_ROC_AUC"]
            },
            "HistGradientBoosting": {
                "mean_AP": train_results["cv_metrics"]["HistGradientBoosting"]["mean_AP"],
                "std_AP": train_results["cv_metrics"]["HistGradientBoosting"]["std_AP"],
                "mean_ROC_AUC": train_results["cv_metrics"]["HistGradientBoosting"]["mean_ROC_AUC"],
                "std_ROC_AUC": train_results["cv_metrics"]["HistGradientBoosting"]["std_ROC_AUC"]
            }
        },
        "held_out_test_evaluation": {
            "selected_model": selected_name,
            "sample_size": len(y_test),
            "AP": eval_selected["AP"],
            "ROC_AUC": eval_selected["ROC_AUC"],
            "metrics_at_default_threshold_0_5": eval_selected["metrics_at_default_threshold_0_5"],
            "cv_selected_alternative_threshold": eval_selected["metrics_at_cv_selected_threshold"],
            "dummy_baseline_reference": eval_dummy
        }
    }
    
    # Prepare test_predictions.csv DataFrame (anonymous y_true, probability, predicted_label only)
    test_preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "predicted_probability": eval_selected["test_probabilities"],
        "predicted_label": eval_selected["test_predictions_05"]
    })
    
    if save_artifacts:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        # 1. artifacts/model.joblib (Confirming identical pipeline object used for test evaluation)
        model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
        joblib.dump(selected_pipeline, model_path)
        
        # 2. artifacts/metrics.json
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
            
        # 3. artifacts/test_predictions.csv
        preds_path = os.path.join(ARTIFACTS_DIR, "test_predictions.csv")
        test_preds_df.to_csv(preds_path, index=False)
        
        print("Saved artifacts successfully:")
        print(f" - {model_path}")
        print(f" - {metrics_path}")
        print(f" - {preds_path}")
        
    return {
        "metrics": metrics_payload,
        "test_predictions": test_preds_df,
        "selected_pipeline": selected_pipeline
    }


if __name__ == "__main__":
    out = run_evaluation(save_artifacts=False)
    selected = out["metrics"]["model_selection"]["selected_model"]
    test_res = out["metrics"]["held_out_test_evaluation"]
    
    print(f"\nFinal Test Results for {selected}:")
    print(f" - AP (Average Precision): {test_res['AP']!r}")
    print(f" - ROC-AUC: {test_res['ROC_AUC']!r}")
    
    m05 = test_res["metrics_at_default_threshold_0_5"]
    print("\n=== Metrics at Default Threshold 0.50 ===")
    print(f"Threshold: {m05['threshold']!r}")
    print(f"Precision: {m05['precision']!r}")
    print(f"Recall: {m05['recall']!r}")
    print(f"F1: {m05['f1']!r}")
    print(f"Confusion Matrix: TP={m05['confusion_matrix']['TP']!r}, FP={m05['confusion_matrix']['FP']!r}, TN={m05['confusion_matrix']['TN']!r}, FN={m05['confusion_matrix']['FN']!r}")
    
    malt = test_res["cv_selected_alternative_threshold"]
    print("\n=== Metrics at CV-Selected Alternative Threshold ===")
    print(f"Exact CV Threshold Selected: {malt['threshold']!r}")
    print(f"Precision: {malt['precision']!r}")
    print(f"Recall: {malt['recall']!r}")
    print(f"F1: {malt['f1']!r}")
    print(f"Confusion Matrix: TP={malt['confusion_matrix']['TP']!r}, FP={malt['confusion_matrix']['FP']!r}, TN={malt['confusion_matrix']['TN']!r}, FN={malt['confusion_matrix']['FN']!r}")

