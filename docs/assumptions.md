# Modeling Decisions, Audit Rules, and Assumptions Log

This document records every non-obvious engineering decision, architectural assumption, parameter choice, and data governance rule established across the European Bank Customer Churn project.

---

### 1. Dataset Integrity & Feature Policy Assumptions

- **Static Single-Snapshot Architecture:**  
  The supplied dataset (`European_Bank.csv`) represents a single cross-sectional snapshot of 10,000 customers. It contains no timestamps, repeated per-customer observations, or time-series index.  
  *Decision:* Strict exclusion of all temporal and longitudinal techniques (e.g., Granger causality, change-point/breakpoint regression, or rolling time trends). Applying time-series methods to static cross-sectional data is methodologically invalid and misleading.
- **Identifier Exclusion (`CustomerId`, `Surname`):**  
  Customer IDs and surnames are personal identifiers. Including them would introduce severe memorization risk, potential data leakage, and ethical violations. They are strictly excluded from the feature matrix $X$.
- **Constant Field Exclusion (`Year = 2025`):**  
  The `Year` column contains a single unique value (2025) across all 10,000 records. With zero variance, it provides zero predictive variance and cannot support time-trend analysis. It is excluded from model training.
- **Strict Target Isolation (`Exited`):**  
  `Exited` is exclusively the classification target $y$. Quality gate assertions verify that `Exited` is never present in $X$ under any transformation step.

---

### 2. Segmentation & Cohort Definitions

- **Age Band Thresholds (`<30`, `30–45`, `46–59`, `60+`):**  
  *Rationale:* Rather than assuming arbitrary continuous linearity, age was discretized into four demographic cohorts reflecting life stages (early career, mid-career/family formation, pre-retirement, and retirement). Crucially, this isolates the `46–59` age bracket, which exhibits a 51.10% observed churn rate (810 exits from 1,585 customers) — 2.51x the portfolio baseline of 20.37%.
- **Categorical Treatment of Product Counts (`NumOfProducts`):**  
  *Rationale:* Customer exit rates do not follow a linear monotonic trend with respect to product holdings:
  - 1 Product: **27.71%** churn rate ($n=5,084$)
  - 2 Products: **7.58%** churn rate ($n=4,590$)
  - 3 Products: **82.71%** churn rate ($n=266$)
  - 4 Products: **100.00%** churn rate ($n=60$)  
  Treating `NumOfProducts` as a continuous numeric feature would force a linear slope, obscuring the dramatic protective effect of holding exactly 2 products and the extreme churn associated with 3+ products. It is therefore one-hot encoded as a nominal categorical variable.
- **Small-Sample Flagging Threshold ($n < 300$):**  
  *Rationale:* Cohorts with denominators smaller than 300 records (specifically Products 3 with $n=266$ and Products 4 with $n=60$) exhibit high sample variance and extreme rates. Displaying "100% churn rate" without visible sample size context would lead stakeholders to over-extrapolate. An explicit caution marker is triggered whenever $n < 300$.

---

### 3. Model Comparison & Selection Protocol

- **Partitioning Discipline (80/20 Stratified Split):**  
  A single stratified 80/20 train/test split (`random_state=42`) was drawn at the outset. The held-out test split (2,000 records) was locked away and never inspected or evaluated during candidate model comparison or threshold tuning.
- **Training-Only 5-Fold Stratified Cross-Validation:**  
  All candidate comparisons were conducted using 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the 8,000 training records.
- **Pipeline and Preprocessor Isolation:**  
  Each estimator was assigned its own independent `Pipeline` and its own `ColumnTransformer`. Preprocessing (median imputation, scaling, one-hot encoding) was fitted strictly inside each training fold to ensure zero data leakage between folds.
  - Logistic Regression received sparse one-hot encoding (`sparse_output=True`, sklearn default).
  - HistGradientBoosting received dense one-hot encoding (`sparse_output=False`) to satisfy scikit-learn's dense array requirement.
- **Primary Selection Metric: Average Precision (AP):**  
  With positive class prevalence at ~20.37%, standard Accuracy is uninformative (a naive model predicting "stay" achieves 79.63% accuracy while catching 0% of churners). ROC-AUC can be overly optimistic under class imbalance. Average Precision (area under the Precision-Recall curve) evaluates precision across all recall thresholds and was chosen as the authoritative selection criterion.
- **The 0.01 AP Interpretability Tie-Breaker Margin:**  
  *Rationale:* Complex non-linear tree ensembles often provide marginal performance gains at the expense of explainability. The protocol mandated that if `HistGradientBoosting` scored within **0.01 AP** of `LogisticRegression`, Logistic Regression would be chosen as the winning architecture for linear transparency.  
  *Outcome:* `HistGradientBoosting` achieved Mean CV AP = **0.6905**, outperforming `LogisticRegression` (**0.6168**) by **+0.0737 AP**. Because this difference exceeds 0.01, HistGradientBoosting was decisively selected.

---

### 4. Decision Threshold Reporting Philosophy

- **Dual Threshold Presentation (0.50 vs. CV-Tuned Alternative):**  
  *Rationale:* At the standard 0.50 threshold, the model achieves 73.1% precision but only 47.4% recall (missing 214 of 407 churners, 71 false alarms). Tuning the threshold to maximize $F_1$ on cross-validated training predictions yields an alternative threshold (0.27), increasing recall to 69.0% (catching 281 churners) while increasing false alarms from 71 to 213 (precision 56.9%).
- **Why Neither Threshold is Picked as "The Answer":**  
  Selecting an operational threshold requires knowing the bank's asymmetric cost function:
  $$\text{Expected Cost} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP}$$
  where $C_{\text{FN}}$ is the business cost of losing an undetected customer, and $C_{\text{FP}}$ is the outreach/marketing cost of contacting a loyal customer. Because the supplied dataset contains no financial figures, customer lifetime values, or campaign cost budgets, declaring one threshold as "optimal" would be intellectually dishonest. Both are presented with exact counts.

---

### 5. Explainability vs. Causality

- **Attribution as Association, Not Causation:**  
  SHAP values and permutation importance metrics measure feature attribution within the model's internal mathematical logic. They indicate which features the algorithm relies on to rank customer risk. They do not constitute causal proof that intervening on a feature (e.g., offering a product discount or balance incentive) will alter customer behavior in the real world.
