"""
train.py
Model training, 5-fold cross-validation, and selection protocol.

Follows the blueprint:
- Stratified 80/20 train/test split (random_state=42).
- 5-fold StratifiedKFold cross-validation on training partition only.
- Compares:
  1) DummyClassifier(strategy="prior")
  2) LogisticRegression(max_iter=2000, random_state=42)
  3) HistGradientBoostingClassifier(random_state=42)
- Separate Pipeline and ColumnTransformer per estimator.
- Sparse one-hot output for LogisticRegression, dense one-hot output for HistGradientBoosting.
- Selection metric: Mean cross-validated Average Precision (AP).
  If the two scores are within 0.01 AP of each other, prefer Logistic Regression for interpretability.
- Selected pipeline is fitted on full training partition only (never on test data).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

from src.data_validation import (
    load_and_validate_data,
    get_features_and_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Builds an isolated Pipeline with its own ColumnTransformer for Logistic Regression.
    Uses sparse one-hot encoding (the sklearn default).
    """
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_tf, NUMERIC_FEATURES),
        ("cat", cat_tf, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ])


def build_hist_gradient_boosting_pipeline() -> Pipeline:
    """
    Builds an isolated Pipeline with its own ColumnTransformer for HistGradientBoosting.
    Uses dense one-hot encoding (sparse_output=False) required by HistGradientBoosting.
    """
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_tf, NUMERIC_FEATURES),
        ("cat", cat_tf, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42))
    ])


def build_dummy_pipeline() -> Pipeline:
    """
    Builds an isolated baseline Pipeline using DummyClassifier(strategy="prior").
    """
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_tf, NUMERIC_FEATURES),
        ("cat", cat_tf, CATEGORICAL_FEATURES)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="prior"))
    ])


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Runs 5-fold StratifiedKFold on training data only.
    Computes fold-by-fold Average Precision (AP) and ROC-AUC.
    Returns CV summary metrics and out-of-fold probability predictions.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    pipeline_builders = {
        "Dummy": build_dummy_pipeline,
        "LogisticRegression": build_logistic_regression_pipeline,
        "HistGradientBoosting": build_hist_gradient_boosting_pipeline
    }
    
    cv_metrics = {}
    oof_predictions = {name: np.zeros(len(y_train)) for name in pipeline_builders}
    
    for name, builder in pipeline_builders.items():
        fold_ap = []
        fold_roc = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            pipe = builder()
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Preprocessing is fitted strictly inside each training fold
            pipe.fit(X_tr, y_tr)
            
            val_probs = pipe.predict_proba(X_va)[:, 1]
            oof_predictions[name][val_idx] = val_probs
            
            ap = average_precision_score(y_va, val_probs)
            roc = roc_auc_score(y_va, val_probs)
            fold_ap.append(ap)
            fold_roc.append(roc)
            
        cv_metrics[name] = {
            "mean_AP": float(np.mean(fold_ap)),
            "std_AP": float(np.std(fold_ap)),
            "mean_ROC_AUC": float(np.mean(fold_roc)),
            "std_ROC_AUC": float(np.std(fold_roc)),
            "fold_AP": [float(x) for x in fold_ap],
            "fold_ROC_AUC": [float(x) for x in fold_roc]
        }
        
    return cv_metrics, oof_predictions


def find_cv_optimal_threshold(y_train: pd.Series, oof_probs: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Finds threshold that maximizes F1 score on cross-validated out-of-fold training predictions.
    Returns: (best_threshold, best_f1, precision, recall)
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = -1.0
    best_th = 0.5
    best_prec = 0.0
    best_rec = 0.0
    
    for th in thresholds:
        preds = (oof_probs >= th).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)
            best_prec = float(precision_score(y_train, preds, zero_division=0))
            best_rec = float(recall_score(y_train, preds, zero_division=0))
            
    return best_th, best_f1, best_prec, best_rec


def train_and_select_model(csv_path: str = "data/raw/European_Bank.csv") -> Dict[str, Any]:
    """
    Executes the training and model selection pipeline:
    1. Loads and audits dataset.
    2. Splits into stratified 80/20 train/test split with random_state=42.
    3. Cross-validates candidates using 5-fold StratifiedKFold on train only.
    4. Applies model selection protocol:
       - Compares mean CV AP between Logistic Regression and HistGradientBoosting.
       - If difference <= 0.01, selects Logistic Regression for interpretability.
       - Otherwise selects the model with higher mean CV AP.
    5. Optimizes threshold on training OOF predictions.
    6. Fits the selected pipeline and dummy baseline on full training partition only.
    """
    df, audit = load_and_validate_data(csv_path)
    X, y = get_features_and_target(df)
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Cross-validation on training data only
    cv_metrics, oof_predictions = cross_validate_candidates(X_train, y_train)
    
    lr_ap = cv_metrics["LogisticRegression"]["mean_AP"]
    hgb_ap = cv_metrics["HistGradientBoosting"]["mean_AP"]
    ap_diff = abs(hgb_ap - lr_ap)
    
    # Model Selection Rule
    if ap_diff <= 0.01:
        selected_model_name = "LogisticRegression"
        selection_rationale = (
            f"LogisticRegression selected: mean CV AP ({lr_ap:.4f}) is within 0.01 of "
            f"HistGradientBoosting ({hgb_ap:.4f}), preferred for linear interpretability."
        )
        selected_builder = build_logistic_regression_pipeline
    else:
        if hgb_ap > lr_ap:
            selected_model_name = "HistGradientBoosting"
            selection_rationale = (
                f"HistGradientBoosting selected: mean CV AP ({hgb_ap:.4f}) exceeds "
                f"LogisticRegression ({lr_ap:.4f}) by {ap_diff:.4f} (> 0.01 threshold)."
            )
            selected_builder = build_hist_gradient_boosting_pipeline
        else:
            selected_model_name = "LogisticRegression"
            selection_rationale = (
                f"LogisticRegression selected: higher mean CV AP ({lr_ap:.4f} vs {hgb_ap:.4f})."
            )
            selected_builder = build_logistic_regression_pipeline
            
    # Find CV-selected alternative threshold on training OOF predictions
    best_th, oof_f1, oof_prec, oof_rec = find_cv_optimal_threshold(
        y_train,
        oof_predictions[selected_model_name]
    )
    
    # Fit winning pipeline on FULL training partition only
    selected_pipeline = selected_builder()
    selected_pipeline.fit(X_train, y_train)
    
    # Fit dummy pipeline on FULL training partition for baseline reference
    dummy_pipeline = build_dummy_pipeline()
    dummy_pipeline.fit(X_train, y_train)
    
    return {
        "selected_model_name": selected_model_name,
        "selection_rationale": selection_rationale,
        "selected_pipeline": selected_pipeline,
        "dummy_pipeline": dummy_pipeline,
        "cv_metrics": cv_metrics,
        "cv_optimal_threshold": {
            "threshold": best_th,
            "f1": oof_f1,
            "precision": oof_prec,
            "recall": oof_rec
        },
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "audit": audit
    }


if __name__ == "__main__":
    results = train_and_select_model()
    print("=== Cross-Validation Results ===")
    for model_name, metrics in results["cv_metrics"].items():
        print(f"{model_name:22s} | Mean AP: {metrics['mean_AP']:.4f} ± {metrics['std_AP']:.4f} | "
              f"Mean ROC-AUC: {metrics['mean_ROC_AUC']:.4f} ± {metrics['std_ROC_AUC']:.4f}")
        fold_str = ", ".join([f"Fold {i+1}: {ap:.4f}" for i, ap in enumerate(metrics["fold_AP"])])
        print(f"   Fold APs: [{fold_str}]")
    print("\nWinner:", results["selected_model_name"])
    print("Rationale:", results["selection_rationale"])
    print("CV-selected alternative threshold:", results["cv_optimal_threshold"])
    
    print("\n=== Raw Full Precision Values ===")
    hgb_metrics = results["cv_metrics"]["HistGradientBoosting"]
    lr_metrics = results["cv_metrics"]["LogisticRegression"]
    print(f"HistGradientBoosting mean_AP: {hgb_metrics['mean_AP']!r}")
    print(f"HistGradientBoosting std_AP: {hgb_metrics['std_AP']!r}")
    print(f"HistGradientBoosting mean_ROC_AUC: {hgb_metrics['mean_ROC_AUC']!r}")
    print(f"HistGradientBoosting std_ROC_AUC: {hgb_metrics['std_ROC_AUC']!r}")
    print(f"LogisticRegression mean_AP: {lr_metrics['mean_AP']!r}")
    print(f"LogisticRegression std_AP: {lr_metrics['std_AP']!r}")
    print(f"LogisticRegression mean_ROC_AUC: {lr_metrics['mean_ROC_AUC']!r}")
    print(f"LogisticRegression std_ROC_AUC: {lr_metrics['std_ROC_AUC']!r}")
    print(f"Raw AP Difference (HGB - LR): {hgb_metrics['mean_AP'] - lr_metrics['mean_AP']!r}")


