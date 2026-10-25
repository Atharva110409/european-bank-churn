# Retrospective Customer Churn Risk Classification: A Leakage-Safe Statistical Audit and Comparative Benchmark on Cross-Sectional European Banking Data

**[Author Name Placeholder]**  
Department of [Department Placeholder], [Institution / University Placeholder]  
[Program / Laboratory Placeholder], [City, Country Placeholder]  
Email: [author.email@placeholder.edu]

---

### Abstract
This study conducts an audited, leakage-safe empirical investigation of customer churn prediction using an educational cross-sectional dataset of 10,000 retail banking accounts across three European geographies (France, Germany, and Spain). We formulate four core research questions addressing multiple-comparison-corrected segment disparities, leak-free supervised model selection, operational threshold trade-offs, and the boundaries of feature attribution. Non-parametric hypothesis tests with Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$ confirm that observed churn disparities across all four analyzed dimensions—Geography ($\chi^2 = 301.26$), Age Band ($\chi^2 = 1206.70$), Product Count ($\chi^2 = 1503.63$), and Active Membership ($\chi^2 = 242.99$)—survive correction with adjusted $p < 10^{-54}$. Specifically, customers in the 46–59 age band exhibit an observed churn rate of **51.10%** (1,000-resample bootstrap 95% CI: [48.77%, 53.63%]), representing a **2.51x** elevation over the portfolio baseline rate of **20.37%** (95% CI: [19.64%, 21.18%]). Under a leak-free 5-fold cross-validation protocol using mean Average Precision (AP), a Histogram-based Gradient Boosting Classifier decisively outperformed regularized Logistic Regression by +0.0737 AP (0.6905 vs. 0.6168), exceeding a pre-registered 0.01 AP margin favoring parsimony. On an untouched held-out test partition ($N = 2,000$), the winning model achieved an AP of **0.6980** (95% CI: [0.6512, 0.7394]) and an ROC-AUC of **0.8578** (95% CI: [0.8360, 0.8799]). Evaluating operating thresholds reveals that the default 0.50 cutoff catches 193/407 churners (47.42% recall, missing 52.58%) with 71 false alarms (73.11% precision), whereas a cross-validation-tuned 0.27 threshold catches 281/407 churners (69.04% recall, missing 30.96%) with 213 false alarms (56.88% precision). We emphasize that all reported associations and feature attributions reflect statistical model reliance in static cross-sectional data, not physical mechanisms or causal claims.

**Index Terms**—Customer churn prediction, statistical audit, data leakage prevention, Benjamini-Hochberg FDR, bootstrap confidence intervals, precision-recall trade-off, SHAP attribution, gradient boosting.

---

## I. Introduction

Customer attrition poses a continuous challenge to retail banking institutions, directly undermining net customer acquisition economics and multi-year customer lifetime value (LTV) [1]. In response, banking analytics teams frequently deploy supervised machine learning models to identify accounts exhibiting elevated risk of departure. However, the translation of academic churn modeling into reliable operational decision-making is fraught with methodological challenges. Three systemic vulnerabilities recur throughout the applied predictive literature:
1. *Cohort over-interpretation without multiple-testing control:* Subgroup analysis frequently highlights extreme churn rates in exploratory segments without adjusting for simultaneous inference or reporting estimation uncertainty, mistaking sample variance for durable patterns.
2. *Methodological and data leakage:* Inappropriate preprocessing—such as fitting feature transformers across entire datasets prior to train-test partitioning—artificially inflates diagnostic metrics, producing optimistically biased evaluations that degrade precipitously in deployment [2], [3].
3. *Conflation of predictive attribution with causal mechanisms:* Interpretability tools like SHAP or permutation importance are often mistakenly treated as actionable causal levers, encouraging unverified business interventions on features that are merely correlated with attrition [4].

To address these vulnerabilities, this paper presents a leakage-safe statistical audit and benchmark evaluation conducted on `European_Bank.csv`, a cross-sectional dataset of 10,000 banking customers. We establish a pre-registered feature policy, apply Benjamini-Hochberg False Discovery Rate (FDR) corrections [5] across cohort tests, derive 1,000-resample non-parametric bootstrap confidence intervals [6], and compare candidate model architectures under training-only cross-validation. Furthermore, we explicitly evaluate operating threshold trade-offs rather than defaulting to arbitrary cutoffs.

To maintain empirical rigor, our investigation is structured around four pre-registered Research Questions (RQs), summarized in Table I.

### TABLE I: Research Questions and Methodological Focus
| Research Question | Analytical Scope | Statistical Methodology |
| :--- | :--- | :--- |
| **RQ1: Cohort Disparities** | Do observed segment-level churn differences survive correction for multiple comparisons? | Contingency tables, Pearson $\chi^2$ test of independence, Cramér's $V$ effect size, Benjamini-Hochberg FDR control ($q=0.05$), and 1,000-resample bootstrap 95% CIs. |
| **RQ2: Predictive Validity** | Can the available features predict churn meaningfully better than a majority baseline under leakage-safe validation? | Stratified 80/20 train/test split, 5-fold cross-validated Average Precision (AP), ROC-AUC, pre-registered tie-breaker rule (+0.01 AP margin), and pipeline encapsulation. |
| **RQ3: Operational Trade-offs** | What precision/recall tradeoff does the selected model's threshold choice impose, and how should that tradeoff be communicated? | Confusion matrix decomposition at default (0.50) vs. CV-tuned (0.27) thresholds, quantifying caught churners, false alarms, and missed attrition. |
| **RQ4: Attribution Boundaries** | What does feature attribution reveal, and what are the limits of treating it as evidence? | TreeExplainer SHAP attribution ($N=1,000$) and held-out test permutation importance ($B=10$), accompanied by formal non-causal boundary delineation. |


---

## II. Related Work

### A. Machine Learning in Customer Attrition
Supervised churn classification in financial services has historically evolved from parametric linear models, such as logistic regression and linear discriminant analysis, toward non-linear tree ensembles, including Random Forests, XGBoost, and LightGBM [1], [7]. While non-linear ensembles consistently demonstrate superior discriminatory ranking, class imbalance (typically 15%–25% positive prevalence) renders standard classification accuracy uninformative. Modern consensus mandates the adoption of threshold-agnostic, ranking-sensitive metrics, specifically Average Precision (the area under the precision-recall curve) and the Receiver Operating Characteristic Area Under the Curve (ROC-AUC) [8].

### B. Data Leakage and Reproducibility Failures
Data leakage occurs when information from outside the training partition influences model training, feature selection, or hyperparameter optimization. Kaufman et al. [2] formalized the taxonomy of data leakage in data mining, distinguishing between target leakage (direct or indirect inclusion of target proxies in feature matrices) and train-test contamination. Recently, Kapoor and Narayanan [3] conducted a systematic survey across 17 fields, demonstrating that data leakage represents a primary driver of the reproducibility crisis in machine-learning-based science, affecting hundreds of published studies. In banking churn settings, common leakage vectors include global scaling across partitions, target-correlated ID variables, and refitting models on combined datasets. Ensuring strict pipeline encapsulation where all transformers are fitted exclusively on training folds is essential for valid empirical claims [9].

### C. Multiple Hypothesis Testing in Segment Audits
Exploratory subgroup segmentation often involves testing numerous candidate customer cohorts against attrition. When conducting multiple significance tests simultaneously, the family-wise error rate escalates, yielding spurious statistical discoveries. Benjamini and Hochberg [5] introduced the False Discovery Rate (FDR) controlling procedure, which guarantees that the expected proportion of false rejections among all rejected null hypotheses remains below a predefined threshold $q$. Integrating Benjamini-Hochberg FDR corrections into cohort audits ensures that reported segment disparities reflect robust statistical associations rather than multi-comparison artifacts.

### D. Model Explainability and Causal Boundaries
The advent of Shapley Additive Explanations (SHAP), grounded in cooperative game theory [4], has transformed model interpretability in tabular machine learning. Permutation feature importance [10] provides an orthogonal, model-agnostic measure of how shuffling a feature degrades global loss. However, both methods measure *statistical reliance* within a specific mathematical functional form; they do not establish that intervening upon a feature will alter the target state in the real world [11]. Methodological honesty requires reporting feature attribution strictly as model introspection rather than causal prescription.

---

## III. Methodology and Dataset Audit

### A. Dataset Architecture and Provenance
The experimental dataset comprises `European_Bank.csv`, containing exactly 10,000 records across 14 tabular fields. The target variable `Exited` denotes whether a customer closed their account:
- **Retained ($y = 0$):** 7,963 accounts (79.63%)
- **Exited ($y = 1$):** 2,037 accounts (20.37%)

*Provenance Disclosure:* The file resembles the widely circulated educational `Churn_Modelling` benchmark, but the exact provenance of this specific CSV has not been independently verified. Describe it as an educational customer-churn dataset supplied for this project. Do not present it as actual European Central Bank data or proprietary records from a European bank.

Data audit procedures (`src/data_validation.py`) confirmed that all 10,000 records contain zero missing values across all columns and zero duplicate `CustomerId` entries. The column `Year` was found to be constant (`2025`) across all 10,000 rows, offering zero statistical variance. Because the dataset contains no repeated longitudinal observations, transaction timestamps, or event-time logs, it represents a static, single-snapshot cross-sectional observation. Temporal methods (e.g., survival hazard modeling, ARIMA forecasting, or Granger causality) are strictly invalid and are excluded from this study.

### B. Segment Statistical Testing
To address RQ1, we audit customer exit patterns across four primary operational dimensions: Geography, Age Band, Number of Products, and Active Membership. For each dimension, we construct contingency tables against `Exited` and compute Pearson's Chi-square test of independence:
$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
where $O_{ij}$ and $E_{ij}$ denote observed and expected cell counts, respectively. To quantify association magnitude independent of sample size, we calculate Cramér's $V$:
$$V = \sqrt{\frac{\chi^2}{N \cdot \min(r - 1, c - 1)}}$$
where $N = 10,000$, and $r$ and $c$ represent row and column dimensions.

To prevent spurious discovery inflation across simultaneous segment evaluations, raw $p$-values are adjusted using the Benjamini-Hochberg [5] step-up procedure at a false discovery rate of $q = 0.05$. Ranks $i \in \{1, \dots, m\}$ (here $m = 4$) are assigned to sorted raw $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, with adjusted values computed as:
$$p_{(i)}^{\text{adj}} = \min \left( \min_{k \ge i} \left\{ \frac{m}{k} p_{(k)} \right\}, 1 \right)$$
ensuring the monotonicity constraint $p_{(i)}^{\text{adj}} \ge p_{(i)}$.

### C. Non-Parametric Bootstrap Confidence Intervals
To avoid relying solely on asymptotic normality assumptions, we construct 1,000-resample non-parametric percentile bootstrap confidence intervals [6] using a fixed pseudo-random seed (`random_seed=42`) via `src/statistics.py`. For a sample statistic $\hat{\theta}$, bootstrap estimates $\theta^{*(1)}, \dots, \theta^{*(B)}$ ($B = 1,000$) are generated by sampling with replacement. The two-sided 95% confidence interval is defined by the empirical 2.5th and 97.5th percentiles:
$$\text{CI}_{0.95} = \left[ \theta^{*}_{(0.025 \cdot B)}, \; \theta^{*}_{(0.975 \cdot B)} \right]$$
We compute bootstrap intervals for four key parameters: (1) portfolio overall churn rate, (2) the 46–59 age band churn rate, (3) final model test Average Precision (AP), and (4) final model test ROC-AUC.

### D. Feature Governance and Leakage Prevention Policy
To eliminate target and entity leakage, candidate features are strictly governed:
1. `Exited` is designated strictly as target $y$ and is defensively excluded from the feature space $X$.
2. `CustomerId` and `Surname` are omitted to prevent high-cardinality entity memorization and privacy infringement.
3. `Year` is omitted due to zero variance.
4. The remaining 10 features form candidate matrix $X$:
   - *Continuous / Numeric (5):* `CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`.
   - *Categorical / Discrete (5):* `Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`.

*Categorical Treatment of Product Holding:* Crucially, `NumOfProducts` is encoded as a discrete categorical feature rather than an integer scale. Exploratory inspection reveals that customer churn does not vary monotonically with product holdings; treating it as continuous would impose an invalid linear constraint, obscuring the protective association of dual-product holdings.

### E. Model Selection and Evaluation Protocol
The dataset is partitioned once into an 80% training split ($N = 8,000$) and a 20% held-out test split ($N = 2,000$) using stratified sampling on `Exited` (`random_state=42`). The held-out test partition is strictly sequestered and accessed only once for final evaluation.

Within the training set, model selection is performed using 5-fold `StratifiedKFold(shuffle=True, random_state=42)`. All preprocessing transformations (median imputation, standard scaling for numeric features, and one-hot encoding for categorical features) are encapsulated within independent scikit-learn `Pipeline` and `ColumnTransformer` instances [9], fitted strictly on each fold's training indices.

Candidate architectures include:
- `DummyClassifier(strategy="prior")`: Predicts the empirical training base rate (majority baseline).
- `LogisticRegression(max_iter=2000, random_state=42)`: $L_2$-regularized linear model with sparse one-hot encoding.
- `HistGradientBoostingClassifier(random_state=42)`: Non-linear histogram-based gradient-boosted decision tree ensemble with dense encoding.

*Explicit Decision Criterion:* Model selection is governed primarily by mean cross-validated Average Precision (AP). To balance empirical performance with parsimony and operational simplicity, a pre-registered decision rule was instituted:
$$\text{Select } \text{LogisticRegression} \quad \text{if} \quad \overline{\text{AP}}_{\text{HGB}} - \overline{\text{AP}}_{\text{LogReg}} \le 0.01$$
$$\text{Select } \text{HistGradientBoosting} \quad \text{if} \quad \overline{\text{AP}}_{\text{HGB}} - \overline{\text{AP}}_{\text{LogReg}} > 0.01$$

---

## IV. Empirical Results

### A. Segment Statistical Audit (RQ1)
Aggregating the 10,000 accounts across the four operational dimensions yields the baseline segment distributions detailed in Table II.

### TABLE II: Cohort Segmentation and Observed Churn Distributions
| Dimension | Subgroup | Total Customers ($n$) | Retained ($y=0$) | Exits ($y=1$) | Observed Churn Rate | Share of Total Exits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Geography** | France | 5,014 | 4,204 | 810 | 16.15% | 39.76% |
| | Germany | 2,509 | 1,695 | 814 | **32.44%** | 39.96% |
| | Spain | 2,477 | 2,064 | 413 | 16.67% | 20.27% |
| **Age Band** | <30 | 1,641 | 1,517 | 124 | 7.56% | 6.09% |
| | 30–45 | 6,248 | 5,292 | 956 | 15.30% | 46.93% |
| | **46–59** | **1,585** | **775** | **810** | **51.10%** | **39.76%** |
| | 60+ | 526 | 379 | 147 | 27.95% | 7.22% |
| **Product Count** | 1 | 5,084 | 3,675 | 1,409 | 27.71% | 69.17% |
| | 2 | 4,590 | 4,242 | 348 | **7.58%** | 17.08% |
| | 3 * | 266 | 46 | 220 | 82.71% | 10.80% |
| | 4 * | 60 | 0 | 60 | 100.00% | 2.95% |
| **Activity** | Inactive | 4,849 | 3,547 | 1,302 | **26.85%** | 63.92% |
| | Active | 5,151 | 4,416 | 735 | 14.27% | 36.08% |
| **Portfolio Total** | Full Sample | 10,000 | 7,963 | 2,037 | **20.37%** | 100.00% |

*\* Small sample notice: $n < 300$. High finite-sample variance warrants caution against unconditional generalization.*

To formally test whether these observed differences represent statistically significant departures from independence rather than random fluctuations, we execute the Chi-square and FDR procedure defined in Section III-B. Results are saved in `reports/segment_significance_tests.csv` and summarized in Table III.

### TABLE III: Segment Independence Significance Tests and Multiple Testing Correction
| Dimension | Chi-Square ($\chi^2$) | d.o.f. | Raw $p$-value | Cramér's $V$ | BH Adj. $p$-value ($q=0.05$) | Survives FDR? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Geography** | 301.26 | 2 | $3.83 \times 10^{-66}$ | 0.1736 | $5.11 \times 10^{-66}$ | **Yes (Significant)** |
| **Age Band** | 1,206.70 | 3 | $2.58 \times 10^{-261}$ | 0.3474 | $5.16 \times 10^{-261}$ | **Yes (Significant)** |
| **NumOfProducts** | 1,503.63 | 3 | $< 10^{-300}$ | 0.3878 | $< 10^{-300}$ | **Yes (Significant)** |
| **IsActiveMember** | 242.99 | 1 | $8.79 \times 10^{-55}$ | 0.1559 | $8.79 \times 10^{-55}$ | **Yes (Significant)** |

All four segment dimensions survive Benjamini-Hochberg FDR correction at $q = 0.05$ with adjusted $p$-values orders of magnitude below standard thresholds. Product holdings ($V = 0.3878$) and Age Band ($V = 0.3474$) exhibit the largest association magnitudes.

Applying the 1,000-resample non-parametric bootstrap procedure (`reports/bootstrap_cis.json`) yields:
- **Portfolio Baseline Churn Rate:** Point estimate = **20.37%**, 95% Bootstrap CI: **[19.64%, 21.18%]**.
- **Age 46–59 Cohort Churn Rate:** Point estimate = **51.10%**, 95% Bootstrap CI: **[48.77%, 53.63%]**.

The lower bound of the 46–59 age band interval (48.77%) sits well above twice the upper bound of the portfolio baseline (21.18%), establishing an observed **2.51x** rate ratio that is statistically robust.

### B. Predictive Model Performance (RQ2, RQ3)
The 5-fold cross-validation performance ladder across candidate estimators on the 8,000 training records is documented in Table IV.

### TABLE IV: 5-Fold Training Cross-Validation Comparison Ladder
| Estimator Pipeline | Architecture Description | Mean CV AP ± Std | Mean CV ROC-AUC ± Std | Model Selection Status |
| :--- | :--- | :--- | :--- | :--- |
| `DummyClassifier(prior)` | Majority Prevalence Baseline | 0.2037 ± 0.0000 | 0.5000 ± 0.0000 | Baseline Reference |
| `LogisticRegression` | Scaled Continuous + One-Hot Linear | 0.6168 ± 0.0181 | 0.8314 ± 0.0116 | Candidate |
| **`HistGradientBoosting`** | Non-Linear Binned Tree Ensemble | **0.6905 ± 0.0270** | **0.8566 ± 0.0110** | **Selected Winner (+0.0737 AP)** |

*Selection Decision:* HistGradientBoosting outperformed Logistic Regression by **+0.0737 AP**, decisively surpassing the pre-registered parsimony margin of 0.01 AP. Consequently, HistGradientBoosting was selected as the final production model.

The winning pipeline was retrained on the full 8,000-record training split and evaluated once against the sequestered 2,000-record held-out test partition. Evaluation results, accompanied by 1,000-resample bootstrap 95% confidence intervals, are reported in Table V.

### TABLE V: Held-Out Test Set Diagnostic Performance ($N = 2,000$)
| Evaluation Metric | Point Estimate | 95% Bootstrap Confidence Interval | Baseline Reference (Dummy) |
| :--- | :--- | :--- | :--- |
| **Average Precision (AP)** | **0.6980** | **[0.6512, 0.7394]** | 0.2035 |
| **ROC-AUC** | **0.8578** | **[0.8360, 0.8799]** | 0.5000 |

To address RQ3, we examine the operational consequences of probability decision threshold selection on the test partition (which contains exactly 407 churners and 1,593 retained accounts). Table VI compares the standard default threshold ($p \ge 0.50$) against the cross-validation-tuned threshold ($p \ge 0.27$).

### TABLE VI: Operational Decision Threshold Trade-Off Table ($N = 2,000$)
| Operational Metric | Default Threshold ($p \ge 0.50$) | CV-Tuned Alternative ($p \ge 0.27$) | Net Operational Difference |
| :--- | :--- | :--- | :--- |
| **Precision** | **73.11%** ($193 / 264$) | **56.88%** ($281 / 494$) | -16.23% |
| **Recall** | **47.42%** ($193 / 407$) | **69.04%** ($281 / 407$) | +21.62% |
| **$F_1$ Score** | **0.5753** | **0.6238** | +0.0485 |
| **Miss Rate (Uncaught Exits)** | **52.58%** ($214 / 407$) | **30.96%** ($126 / 407$) | -21.62% (88 fewer missed) |
| **True Positives (Caught Churn)** | 193 | 281 | +88 accounts caught |
| **False Negatives (Missed Churn)** | 214 | 126 | -88 accounts missed |
| **False Positives (False Alarms)** | 71 | 213 | +142 unnecessary contacts |
| **True Negatives (Correct Inaction)** | 1,522 | 1,380 | -142 undisturbed accounts |

### C. Feature Attribution and Reliance (RQ4)
To audit internal model mechanisms, we combine global TreeExplainer SHAP attribution on a representative sample ($N=1,000$) with 10-repeat permutation importance on the held-out test split (`reports/permutation_importance.csv`).

SHAP summary analysis indicates:
1. `Age`: Strong positive attribution. Elevated customer age shifts log-odds predictions positively toward exit, aligning with the 51.10% churn observed in the 46–59 cohort.
2. `NumOfProducts`: Dual-product holding (`NumOfProducts = 2`) produces substantial negative attributions (protective against churn), whereas single-product holdings and 3+ holdings shift attribution toward exit.
3. `IsActiveMember`: Active status (`IsActiveMember = 1`) consistently produces negative attributions toward retention.

Table VII summarizes the model-agnostic test permutation importance rankings.

### TABLE VII: Test Set Permutation Feature Importance ($B = 10$ Repeats)
| Rank | Feature Dimension | Mean AP Degradation | Std Error | Attribution Interpretation Notice |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `NumOfProducts` | **0.2727** | 0.0159 | Predictive reliance, not causal importance |
| 2 | `Age` | **0.2381** | 0.0115 | Predictive reliance, not causal importance |
| 3 | `IsActiveMember` | **0.0949** | 0.0083 | Predictive reliance, not causal importance |
| 4 | `Balance` | **0.0612** | 0.0074 | Predictive reliance, not causal importance |
| 5 | `Geography` | **0.0485** | 0.0041 | Predictive reliance, not causal importance |
| 6 | `Gender` | 0.0081 | 0.0028 | Predictive reliance, not causal importance |
| 7 | `Tenure` | 0.0043 | 0.0042 | Predictive reliance, not causal importance |
| 8 | `CreditScore` | 0.0022 | 0.0050 | Predictive reliance, not causal importance |
| 9 | `HasCrCard` | 0.0018 | 0.0008 | Predictive reliance, not causal importance |
| 10 | `EstimatedSalary` | -0.0027 | 0.0016 | Predictive reliance, not causal importance |

*Attribution Caution:* In alignment with modern interpretability governance, these rankings describe mathematical model reliance within the fitted functional form. They do not describe physical or causal mechanisms.

---

## V. Discussion

### A. RQ1: Segment Disparities and Statistical Correction
*Empirical Assessment:* Bivariate segment comparisons in banking analytics are often reported without adjustment for simultaneous inference. Our audit confirms that all four examined operational dimensions—Geography, Age Band, Product Count, and Active Membership—exhibit statistically significant departures from independence under Pearson's Chi-square tests that comfortably survive Benjamini-Hochberg FDR correction at $q = 0.05$. In particular, customers aged 46–59 display an observed exit rate of 51.10% (95% CI: [48.77%, 53.63%]), representing a 2.51x elevation relative to the portfolio baseline (20.37%). Similarly, dual-product holders exhibit an exit rate of only 7.58%, compared to 27.71% for single-product holders.

*What the evidence establishes:* Observed churn differences across the four dimensions are statistically genuine associations that cannot be explained by random sampling variation under the empirical distribution of this dataset.

*What the evidence does NOT establish:* The evidence does not establish that age or geography *causes* customer attrition, nor that cross-selling a second banking product will causally reduce a customer's exit propensity. Such conclusions confuse statistical association with treatment effects.

### B. RQ2: Predictive Validity Under Leakage-Safe Protocols
*Empirical Assessment:* Under strict leakage-safe cross-validation and pipeline encapsulation, the candidate features demonstrate genuine predictive power. HistGradientBoosting achieved a 5-fold cross-validated AP of 0.6905 ± 0.0270, exceeding regularized Logistic Regression (0.6168 ± 0.0181) by +0.0737 AP. On the untouched held-out test split, the model achieved an AP of 0.6980 (95% CI: [0.6512, 0.7394]) and an ROC-AUC of 0.8578 (95% CI: [0.8360, 0.8799]), vastly outperforming the naive majority baseline (AP 0.2035, ROC-AUC 0.5000).

*What the evidence establishes:* Non-linear gradient boosting extracts substantial non-linear interactions from the ten available features, providing a discriminatory ranking of customer risk that is statistically superior to linear baselines and reproducible under strict leakage-free protocols.

*What the evidence does NOT establish:* The model does not establish prospective forecasting capability over time. Because the underlying data represents a static cross-sectional snapshot without timestamps, the model establishes retrospective discrimination within a single observation window, not longitudinal stability.

### C. RQ3: Operational Thresholds and Precision-Recall Tension
*Empirical Assessment:* The choice of operating threshold exposes an unavoidable operational trade-off that machine learning optimization cannot unilaterally resolve. At the default 0.50 cutoff, the model exhibits strong precision (73.11%) and generates only 71 false alarms across 1,593 retained customers. However, it misses 214 out of 407 actual churners—a false negative miss rate of **52.58%**. Shifting to the CV-tuned threshold of 0.27 increases recall from 47.42% to 69.04% (capturing 88 additional churners), but inflates false alarms from 71 to 213 (a 200% increase), reducing precision to 56.88%.

*What the evidence establishes:* Operating thresholds impose distinct cost burdens: the 0.50 threshold incurs substantial opportunity costs via missed departures, whereas the 0.27 threshold incurs outreach expenses and customer contact fatigue.

*What the evidence does NOT establish:* The analysis does not establish which threshold the bank *should* adopt. Deciding between 0.50 and 0.27 requires institutional financial parameters—specifically the unit cost of marketing outreach versus the net present value of a preserved account—which are absent from the dataset.

### D. RQ4: Interpretability, Feature Reliance, and the Causal Boundary
*Empirical Assessment:* Global SHAP attribution and permutation importance identify product count (`NumOfProducts`, AP drop 0.2727) and age (`Age`, AP drop 0.2381) as the predominant predictive features, followed by active membership (`IsActiveMember`, AP drop 0.0949) and balance (`Balance`, AP drop 0.0612). Income (`EstimatedSalary`) and credit score (`CreditScore`) contribute negligible predictive utility.

*What the evidence establishes:* The fitted gradient boosting model relies overwhelmingly on account product depth, age brackets, and engagement status to generate risk rankings.

*What the evidence does NOT establish:* These rankings do not establish that altering feature values will alter customer behavior. For example, manually discounting fees or bundling accounts to increase product count cannot be inferred to prevent churn; such interventions may backfire if multiple product holdings in dissatisfied customers reflect administrative complexity rather than brand loyalty.

---

## VI. Threats to Validity and Limitations

1. *Static Cross-Sectional Data Structure:* The dataset provides a single static snapshot with a constant `Year` (2025). There are no longitudinal customer histories, rolling activity windows, or event timestamps. Consequently, hazard modeling, survival forecasting, and drift detection are impossible.
2. *Unverified Data Provenance:* The data exhibits structural symmetry with the widely circulated educational `Churn_Modelling` dataset. Because its institutional origins cannot be independently verified, findings should not be interpreted as official European Central Bank data or representative of specific European commercial banks.
3. *Substantial Uncaught Churn (The Real Recall Gap):* Even under the tuned 0.27 threshold, the model fails to detect **30.96% of all churning accounts** (126 missed churners). At the standard 0.50 threshold, the model misses **52.58% of all churning accounts** (214 missed churners). In an operational setting, reliance on this model leaves a substantial portion of customer attrition unaddressed.
4. *Absence of Financial Loss and Unit Margin Data:* The dataset contains no figures for customer profitability, account maintenance margin, customer acquisition cost (CAC), or retention campaign cost. Without an empirical cost matrix, automated optimization of decision thresholds is mathematically ill-posed.
5. *Non-Causal Boundary:* The study relies entirely on observational data. Unobserved confounders (e.g., service outages, competitive market offerings, local branch closures) likely drive both customer product selection and attrition risk.

---

## VII. Conclusion and Future Work

This study has delivered an audited, leakage-safe empirical evaluation of customer churn classification on an educational European banking dataset. By embedding all preprocessing inside pipeline wrappers, adjusting cohort significance tests using Benjamini-Hochberg FDR control ($q=0.05$), and computing 1,000-resample non-parametric bootstrap intervals, we established that cohort disparities—including a 51.10% churn rate in the 46–59 age band—are statistically genuine within this dataset. Furthermore, HistGradientBoosting achieved validated ranking performance (test AP 0.6980, ROC-AUC 0.8578), decisively outperforming linear baselines while exposing the critical trade-off between missing 52.58% of churners at threshold 0.50 versus generating 213 false alarms at threshold 0.27.

To extend this work toward production viability, future efforts should prioritize three concrete objectives:
1. *Acquisition of Longitudinal / Timestamped Data:* Sourcing transactional banking telemetry across repeated temporal intervals to enable genuine survival forecasting and hazard rate estimation.
2. *Empirical Cost-Benefit Optimization:* Incorporating explicit customer lifetime value (LTV) and intervention cost schedules to transition from heuristic thresholding to cost-minimizing Bayes decision boundaries.
3. *Prospective Randomized Intervention Trials:* Conducting randomized controlled trials (A/B testing) on proactive retention campaigns to test whether targeted interventions causally alter customer attrition.

---

## Acknowledgment
The authors acknowledge the open-source scientific Python community for providing the computational tools that enabled this study, specifically `scikit-learn`, `scipy`, `pandas`, `numpy`, and `shap`.

---

## References

[1] A. De Caigny, K. Coussement, and K. W. De Bock, "A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees," *European Journal of Operational Research*, vol. 269, no. 2, pp. 760–772, 2018.

[2] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Transactions on Knowledge Discovery from Data (TKDD)*, vol. 6, no. 4, pp. 1–21, 2012.

[3] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023.

[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS 30)*, 2017, pp. 4765–4774.

[5] Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: A practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 57, no. 1, pp. 289–300, 1995.

[6] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY: Chapman & Hall/CRC, 1993.

[7] Y. Xie, X. Li, E. W. T. Ngai, and W. Ying, "Customer churn prediction using improved balanced random forests," *Expert Systems with Applications*, vol. 36, no. 3, pp. 5445–5449, 2009.

[8] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, p. e0118432, 2015.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[10] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[11] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge, UK: Cambridge University Press, 2009.
