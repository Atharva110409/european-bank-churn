# European Bank Churn Analytics — Submission Summary

This project delivers a reproducible, end-to-end customer churn analytics Minimum Viable Product (MVP) built from the `European_Bank.csv` benchmark. It demonstrates rigorous data validation without silent mutation, multi-dimensional customer segmentation across geography, demographics, product depth, and engagement, and a leak-free machine-learning workflow evaluated on cross-validation and an untouched held-out test split, presented through a responsive three-page Streamlit dashboard.

---

### Dataset Provenance Statement (Verbatim)

> The file resembles the widely circulated educational Churn_Modelling benchmark, but the exact provenance of this specific CSV has not been independently verified. Describe it as an educational customer-churn dataset supplied for this project. Do not present it as actual European Central Bank data or proprietary records from a European bank.

---

### Model Selection Outcome

Candidate classifiers were compared strictly using 5-fold `StratifiedKFold` cross-validation on the 80% training partition (8,000 records) evaluated on Average Precision (AP):
- **DummyClassifier (prior baseline):** Mean CV AP = `0.2037 ± 0.0000` (ROC-AUC: `0.5000 ± 0.0000`)
- **LogisticRegression (max_iter=2000):** Mean CV AP = `0.6168 ± 0.0181` (ROC-AUC: `0.8314 ± 0.0116`)
- **HistGradientBoostingClassifier:** Mean CV AP = `0.6905 ± 0.0270` (ROC-AUC: `0.8566 ± 0.0110`)

**Winner:** `HistGradientBoostingClassifier` was selected because its **+0.0737 AP** advantage over Logistic Regression clearly exceeds the protocol's 0.01 margin where Logistic Regression would have been chosen for interpretability. The winning model was fitted on the full training set and achieved a final held-out test AP of **0.6980** (ROC-AUC: **0.8578**) on 2,000 unseen customer records.

---

### Operating Threshold Trade-Off (Open Business Decision)

At the default **0.50 threshold**, the model catches **193 of 407 actual churners (47.4% recall)** while flagging relatively few false alarms (**71**). At the alternative **0.27 threshold**, it catches **281 of 407 actual churners (69.0% recall)**, but at the cost of more false alarms (**213**). Neither threshold is automatically correct — the right choice depends entirely on whether missing a churning customer or unnecessarily contacting a loyal customer is more costly to the bank's bottom line. Because this dataset contains no financial cost or intervention figures, this MVP explicitly surfaces both thresholds as an open operational choice rather than prescribing an arbitrary business policy.

---

### Automated Quality Gate Verification (8/8 Tests Passing)

- **`test_feature_policy_no_leakage`:** Asserts `Exited` is never in feature matrix `X`, and `CustomerId`, `Surname`, and constant `Year` are strictly excluded from modeling.
- **`test_segment_group_counts_sum_to_full_dataset`:** Asserts group counts across Geography, Age Bands, Products, and Activity sum to exactly 10,000 records and 2,037 exits.
- **`test_zero_result_filter_handling`:** Asserts empty filtered subsets return zero counts and 0.0 rates without raising `ZeroDivisionError` or crashing.
- **`test_artifact_presence_and_schema`:** Asserts `model.joblib`, `metrics.json` (explicitly keyed as `"AP"`), and anonymous `test_predictions.csv` exist with exact schemas.
- **`test_saved_model_exact_reproducibility`:** Asserts loaded `model.joblib` reproduces the saved test probabilities with zero numerical divergence, verifying no silent refit occurred.
- **`test_app_pages_startup`:** Asserts via Streamlit `AppTest` that `app.py`, `pages/1_Segments.py`, `pages/2_Model_Performance.py`, and `pages/3_Explainability.py` compile and execute with zero unhandled exceptions.
- **`test_explainability_artifacts_and_properties`:** Asserts global SHAP plot and permutation importance tables exist with explicit non-causal interpretability notices.
- **`test_statistical_audit_properties`:** Asserts all 4 Chi-square tests of independence, Benjamini-Hochberg FDR adjustments, and 1,000-resample bootstrap 95% CIs satisfy ordering and schema bounds.

---

### Core Limitations

1. **Retrospective Classifier, Not Forecasting:** Classifies historical outcomes from a static snapshot; cannot forecast *when* a customer will churn.
2. **Substantial Churn Omission:** Misses 52.6% of actual churners at default threshold 0.50 (47.4% recall) and still misses 31.0% at alternative threshold 0.27 (69.0% recall).
3. **Correlations, Not Causality:** Feature associations (such as age 46–59 churn concentration or German customer churn) are descriptive correlations, not causal proof.
4. **No Financial or Intervention Data:** Lacks balance turnover, customer lifetime value, or campaign cost metrics; cannot support revenue-at-risk or marketing ROI calculations.
5. **No Production Validation:** Technical benchmark only; has not undergone algorithmic fairness audits, subgroup parity checks, or live banking compliance review.

---

### How to Run

```bash
# 1. Clone repository and navigate to root
cd european-bank-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reproduce pipeline and evaluate artifacts
python -m src.data_validation && python -m src.train && python -m src.evaluate

# 4. Run automated test suite
python -m unittest discover tests

# 5. Launch interactive dashboard
streamlit run app.py
```
