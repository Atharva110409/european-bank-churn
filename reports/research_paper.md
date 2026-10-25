# Leakage-Safe Customer Churn Prediction: A Statistical Audit and Comparative Benchmark

Atharva Gangurde  
B.Tech. Candidate in Artificial Intelligence & Data Science  
MVP Samaj’s Karmaveer Adv. Baburao Ganpatrao Thakare College of Engineering  
Nashik, Maharashtra, India  
Email: atharvagangurde12345@gmail.com  

---

### Abstract
This study conducts an audited, leakage-safe empirical investigation of customer churn prediction using an educational cross-sectional dataset of 10,000 retail banking accounts with geography labels for France, Germany, and Spain. We formulate four research questions addressing multiple-comparison-corrected segment disparities, leakage-free supervised model selection, operational threshold trade-offs, and the boundaries of feature attribution. Pearson chi-square tests with Benjamini–Hochberg false discovery rate (FDR) control at $q = 0.05$ indicate that observed churn disparities across all four analyzed dimensions—geography ($\chi^2 = 301.26$), age band ($\chi^2 = 1206.70$), product count ($\chi^2 = 1503.63$), and active membership ($\chi^2 = 242.99$)—survive correction, with adjusted $p$-values below $10^{-54}$. Customers in the 46–59 age band exhibit an observed churn rate of 51.10% (1,000-resample bootstrap 95% confidence interval (CI): [48.77%, 53.63%]), or 2.51 times the portfolio baseline of 20.37% (95% CI: [19.64%, 21.18%]). Under a leakage-safe five-fold cross-validation protocol using mean average precision (AP), a histogram-based gradient boosting classifier outperformed regularized logistic regression by 0.0737 AP (0.6905 versus 0.6168), exceeding a prespecified 0.01 AP margin favoring parsimony. On a held-out test partition ($N = 2,000$), the selected model achieved AP of 0.6980 (95% CI: [0.6512, 0.7394]) and ROC-AUC of 0.8578 (95% CI: [0.8360, 0.8799]). At the default 0.50 threshold, the model identifies 193 of 407 churners (47.42% recall) with 71 false alarms (73.11% precision). A cross-validation-tuned threshold of 0.27 identifies 281 churners (69.04% recall) but produces 213 false alarms (56.88% precision). All reported associations and feature attributions describe statistical patterns in static cross-sectional data; they do not establish physical mechanisms or causal effects.

**Index Terms**—Customer churn prediction, statistical audit, data leakage prevention, Benjamini–Hochberg FDR, bootstrap confidence intervals, precision–recall trade-off, SHAP attribution, gradient boosting.

---

## I. Introduction

Customer attrition is a persistent challenge for retail banking institutions because it can undermine customer-acquisition economics and multi-year customer lifetime value. Supervised machine-learning models are frequently used to identify accounts with elevated departure risk. Translating churn-model results into reliable operational decisions, however, presents methodological challenges.

First, cohort analyses can overstate subgroup differences when exploratory segment comparisons are reported without multiple-testing correction or uncertainty estimates. Second, inappropriate preprocessing—for example, fitting transformations before the train–test split—can leak information across partitions and inflate reported performance. Third, model-interpretability tools such as SHapley Additive exPlanations (SHAP) or permutation importance may be misinterpreted as evidence that changing a feature will change customer behavior [1], [2].

This paper presents a leakage-safe statistical audit and comparative benchmark using `European_Bank.csv`, an educational cross-sectional dataset of 10,000 customer records. The analysis applies Benjamini–Hochberg FDR correction to cohort tests [3], computes 1,000-resample non-parametric bootstrap confidence intervals [4], and compares candidate predictive models using preprocessing pipelines fitted only within training folds. Decision-threshold trade-offs are evaluated explicitly rather than relying only on the default 0.50 cutoff.

The study addresses four research questions (RQs): RQ1, whether segment-level churn disparities survive correction for multiple comparisons; RQ2, whether the available features predict churn better than a prevalence baseline under leakage-safe validation; RQ3, what precision–recall trade-off results from alternative operating thresholds; and RQ4, what feature attribution reveals and where its evidential limits lie.

---

## II. Related Work

### A. Machine Learning for Customer Attrition
Churn-classification methods in financial services range from parametric linear models, such as logistic regression, to non-linear tree ensembles, including random forests and boosting methods [5]–[7]. Because churn datasets often have imbalanced classes, accuracy alone may obscure the ability to rank customers who are likely to leave. Average precision and receiver operating characteristic area under the curve (ROC-AUC) provide complementary ranking-oriented measures; precision–recall analysis is particularly informative for imbalanced data [8].

### B. Data Leakage and Reproducibility
Data leakage occurs when information outside a training partition influences feature construction, selection, model fitting, or optimization. Leakage can arise through target proxies, high-cardinality entity identifiers, or preprocessing fitted using all available records. Kaufman et al. formalized leakage mechanisms and avoidance strategies [9]; later work connected leakage to reproducibility problems in machine-learning research [10]. Encapsulating preprocessing and estimation in a cross-validation pipeline is therefore central to obtaining defensible evaluation estimates [11].

### C. Multiple Testing and Uncertainty
Exploratory segment analyses often test numerous cohorts simultaneously. Without correction, the probability of false discoveries rises as more hypotheses are tested. The Benjamini–Hochberg procedure controls the expected false discovery rate among rejected hypotheses at a selected level [3]. Bootstrap confidence intervals provide a complementary way to express uncertainty without relying solely on asymptotic normal approximations [4].

### D. Explainability and Causal Boundaries
SHAP values allocate a model prediction among features according to a game-theoretic framework [1]. Permutation importance measures how model performance changes when a feature is shuffled. These techniques describe reliance of a fitted model on observed features. They do not, by themselves, identify causal effects or establish that an intervention on a feature would change churn [2].

---

## III. Methodology and Dataset Audit

### A. Dataset Architecture and Provenance
The dataset contains 10,000 records and 14 tabular fields. The target variable `Exited` denotes whether a customer closed their account: 7,963 retained accounts (79.63%) and 2,037 exited accounts (20.37%). Audit procedures reported zero missing values and zero duplicate `CustomerId` entries. The field `Year` is constant at 2025 and therefore contains no statistical variance.

The file resembles the widely circulated educational Churn Modelling benchmark, but the exact provenance of this CSV has not been independently verified. Accordingly, it is described only as an educational customer-churn dataset supplied for this project, not as official European Central Bank data or verified proprietary records from a European bank. Because it contains no repeated longitudinal observations, event timestamps, or transaction logs, it is a static cross-sectional snapshot. Temporal forecasting, survival-hazard modeling, and other longitudinal claims are outside the scope of this analysis.

### B. Segment Statistical Testing (RQ1)
Four operational dimensions are audited: geography, age band, number of products, and active membership. For each dimension, a contingency table is constructed against `Exited`, and Pearson’s chi-square test of independence is calculated:
$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad (1)$$
where $O_{ij}$ and $E_{ij}$ are observed and expected cell counts. Association magnitude is summarized using Cramér’s $V$:
$$V = \sqrt{\frac{\chi^2}{N \min(r - 1, c - 1)}}, \quad (2)$$
where $N = 10,000$ and $r$ and $c$ are the dimensions of the contingency table. Raw $p$-values are adjusted with the Benjamini–Hochberg step-up procedure at FDR level $q = 0.05$. If $p_{(1)} \le \dots \le p_{(m)}$ are the sorted raw $p$-values, the adjusted values are obtained from the monotone step-up estimates, conventionally expressed as
$$p_{(i)}^{\text{adj}} = \min \left( 1, \min_{k \ge i} \frac{m}{k} p_{(k)} \right). \quad (3)$$

### C. Bootstrap Confidence Intervals
Non-parametric percentile bootstrap confidence intervals are estimated with 1,000 resamples and a fixed random seed of 42. For statistic $\hat{\theta}$, bootstrap estimates $\hat{\theta}^{(1)}, \dots, \hat{\theta}^{(B)}$ are generated by sampling with replacement ($B = 1,000$). The two-sided 95% interval is defined by the empirical 2.5th and 97.5th percentiles:
$$\text{CI}_{0.95} = \left[ \hat{\theta}_{(0.025B)}, \hat{\theta}_{(0.975B)} \right]. \quad (4)$$
Intervals are reported for the portfolio churn rate, the churn rate in the 46–59 age band, final-model test AP, and final-model test ROC-AUC.

### D. Feature Governance and Leakage Prevention
The target `Exited` is excluded from the candidate feature matrix. `CustomerId` and `Surname` are excluded to avoid high-cardinality entity memorization and unnecessary exposure of identifying information. `Year` is omitted because it is constant. The remaining ten features comprise five numeric or continuous fields (`CreditScore`, `Age`, `Tenure`, `Balance`, and `EstimatedSalary`) and five categorical or discrete fields (`Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, and `IsActiveMember`).

`NumOfProducts` is treated as a categorical feature rather than a continuous integer scale because the observed churn pattern is not monotonic with product count. Encoding it as continuous would impose a linear ordering that may obscure the low observed churn rate among customers holding two products.

### E. Model Selection and Evaluation (RQ2)
The data are split once into a stratified 80% training partition ($N = 8,000$) and a 20% held-out test partition ($N = 2,000$), using `random_state=42`. The held-out test data are reserved for final evaluation. Model selection within the training partition uses five-fold stratified cross-validation with shuffling and random seed 42. Median imputation, numeric standardization, and categorical one-hot encoding are encapsulated in scikit-learn Pipeline and ColumnTransformer objects, fitted separately within each training fold [11].

Three candidate models are compared: a prior-probability dummy classifier, L2-regularized logistic regression (maximum 2,000 iterations), and histogram-based gradient boosting. Mean cross-validated AP is the primary selection metric. Under the prespecified parsimony rule, logistic regression is selected if the gradient-boosting improvement in mean AP is no greater than 0.01; otherwise, histogram-based gradient boosting is selected:
$$\text{Select logistic regression if } \overline{\text{AP}}_{\text{HGB}} - \overline{\text{AP}}_{\text{LR}} \le 0.01; \text{ otherwise select HGB.} \quad (5)$$

### F. Operating Thresholds and Feature Attribution (RQ3–RQ4)
The selected model is evaluated at a default probability threshold of 0.50 and a cross-validation-tuned alternative of 0.27. We report precision, recall, $F_1$, false positives, false negatives, and true negatives to make the operational trade-off explicit. Feature attribution combines TreeExplainer SHAP values on a representative sample ($N = 1,000$) with test-set permutation importance over ten repeats. Both are interpreted as measures of fitted-model reliance, not causal importance.

---

## IV. Empirical Results

### A. Cohort Distribution and Statistical Audit (RQ1)
Table I summarizes the observed churn distribution. The overall portfolio churn rate is 20.37%. The 46–59 age band has a churn rate of 51.10% (810 of 1,585 customers), whereas customers younger than 30 have a rate of 7.56%. Product count also shows marked heterogeneity: churn is 7.58% among customers with two products, 27.71% among those with one product, and higher in the small groups with three or four products. Estimates for groups with fewer than 300 customers should be interpreted cautiously because of higher finite-sample variability.

Pearson chi-square tests and FDR corrections are reported in Table II. All four dimensions survive Benjamini–Hochberg correction at $q = 0.05$. Product count ($V = 0.3878$) and age band ($V = 0.3474$) have the largest reported association magnitudes. The age-band result is also reflected in uncertainty intervals: the 46–59 cohort churn estimate is 51.10% (95% bootstrap CI: [48.77%, 53.63%]) compared with 20.37% for the full portfolio (95% bootstrap CI: [19.64%, 21.18%]). The point-estimate ratio is 2.51.

### TABLE I: Cohort Segmentation and Observed Churn Distributions
| Dimension | Subgroup | Total | Retained | Exits | Churn rate | Share of exits |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Geography** | France | 5,014 | 4,204 | 810 | 16.15% | 39.76% |
| | Germany | 2,509 | 1,695 | 814 | 32.44% | 39.96% |
| | Spain | 2,477 | 2,064 | 413 | 16.67% | 20.27% |
| **Age band** | < 30 | 1,641 | 1,517 | 124 | 7.56% | 6.09% |
| | 30–45 | 6,248 | 5,292 | 956 | 15.30% | 46.93% |
| | 46–59 | 1,585 | 775 | 810 | 51.10% | 39.76% |
| | 60+ | 526 | 379 | 147 | 27.95% | 7.22% |
| **Product count** | 1 | 5,084 | 3,675 | 1,409 | 27.71% | 69.17% |
| | 2 | 4,590 | 4,242 | 348 | 7.58% | 17.08% |
| | 3* | 266 | 46 | 220 | 82.71% | 10.80% |
| | 4* | 60 | 0 | 60 | 100.00% | 2.95% |
| **Activity** | Inactive | 4,849 | 3,547 | 1,302 | 26.85% | 63.92% |
| | Active | 5,151 | 4,416 | 735 | 14.27% | 36.08% |
| **Portfolio total** | Full sample | 10,000 | 7,963 | 2,037 | 20.37% | 100.00% |

*\*Group size below 300; interpret these rates cautiously.*

### TABLE II: Segment Independence Tests and Multiple-Testing Correction
| Dimension | $\chi^2$ | d.f. | Cramér's $V$ | Adj. $p$ | FDR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Geography** | 301.26 | 2 | 0.1736 | $5.11 \times 10^{-66}$ | Yes |
| **Age band** | 1206.70 | 3 | 0.3474 | $5.16 \times 10^{-261}$ | Yes |
| **Product count** | 1503.63 | 3 | 0.3878 | $< 10^{-300}$ | Yes |
| **Active membership** | 242.99 | 1 | 0.1559 | $8.79 \times 10^{-55}$ | Yes |

### B. Predictive Model Performance (RQ2)
Table III reports five-fold cross-validation results. Histogram-based gradient boosting achieved mean AP of 0.6905 (standard deviation 0.0270), compared with 0.6168 (standard deviation 0.0181) for logistic regression. The difference of 0.0737 exceeds the prespecified 0.01 AP margin; gradient boosting was therefore selected. The held-out evaluation in Table IV reports AP of 0.6980 (95% bootstrap CI: [0.6512, 0.7394]) and ROC-AUC of 0.8578 (95% bootstrap CI: [0.8360, 0.8799]). The dummy reference has AP 0.2035 and ROC-AUC 0.5000.

### TABLE III: Five-Fold Training Cross-Validation Comparison
| Estimator | Mean AP $\pm$ SD | ROC-AUC $\pm$ SD | Decision |
| :--- | :--- | :--- | :--- |
| **Dummy prior** | 0.2037 $\pm$ 0.0000 | 0.5000 $\pm$ 0.0000 | Baseline |
| **Logistic regression** | 0.6168 $\pm$ 0.0181 | 0.8314 $\pm$ 0.0116 | Candidate |
| **Histogram-based gradient boosting** | **0.6905 $\pm$ 0.0270** | **0.8566 $\pm$ 0.0110** | **Selected** |

### TABLE IV: Held-Out Test Performance ($N = 2,000$)
| Metric | Estimate | 95% bootstrap CI |
| :--- | :--- | :--- |
| **Average precision** | **0.6980** | [0.6512, 0.7394] |
| **ROC-AUC** | **0.8578** | [0.8360, 0.8799] |

*Dummy baseline: AP = 0.2035; ROC-AUC = 0.5000.*

### C. Operating Threshold Trade-Offs (RQ3)
Table V compares the default 0.50 cutoff with the tuned 0.27 alternative on the held-out test partition, which contains 407 churners and 1,593 retained accounts. Reducing the threshold increases recall from 47.42% to 69.04%, identifying 88 additional churners and reducing missed churn from 214 to 126. The cost is an increase in false alarms from 71 to 213 and a decrease in precision from 73.11% to 56.88%. The data do not determine which threshold is preferable without institution-specific intervention costs and customer-value estimates.

### TABLE V: Operational Decision Threshold Trade-Off ($N = 2,000$)
| Metric | Default threshold ($p \ge 0.50$) | CV-tuned threshold ($p \ge 0.27$) | Difference |
| :--- | :--- | :--- | :--- |
| **Precision** | 73.11% (193/264) | 56.88% (281/494) | −16.23 percentage points |
| **Recall** | 47.42% (193/407) | 69.04% (281/407) | +21.62 percentage points |
| **$F_1$ score** | 0.5753 | 0.6238 | +0.0485 |
| **Miss rate** | 52.58% (214/407) | 30.96% (126/407) | −21.62 percentage points |
| **True positives (caught churn)** | 193 | 281 | +88 accounts |
| **False negatives (missed churn)** | 214 | 126 | −88 accounts |
| **False positives (false alarms)** | 71 | 213 | +142 contacts |
| **True negatives (correct inaction)** | 1,522 | 1,380 | −142 accounts |

### D. Feature Attribution and Model Reliance (RQ4)
TreeExplainer SHAP summary analysis indicates that age, product count, and active-membership status have important influence on the model’s risk scores. Increased age shifts predictions toward exit in the fitted model. Two-product holding and active-membership status tend to shift attributions toward retention, while single-product and three-or-more-product holdings tend to shift attributions toward exit. These are patterns in the fitted model, not evidence that changing product count or activity would cause churn to decrease.

Table VI shows test-set permutation importance measured by AP degradation over ten repeats. Product count has the largest reported impact (0.2727), followed by age (0.2381), active membership (0.0949), balance (0.0612), and geography (0.0485). Estimated salary has a slightly negative mean degradation (-0.0027), indicating no useful contribution under this permutation procedure and fitted model.

### TABLE VI: Test-Set Permutation Feature Importance (10 Repeats)
| Rank | Feature | Mean AP drop | Std. error |
| :--- | :--- | :--- | :--- |
| 1 | `NumOfProducts` | 0.2727 | 0.0159 |
| 2 | `Age` | 0.2381 | 0.0115 |
| 3 | `IsActiveMember` | 0.0949 | 0.0083 |
| 4 | `Balance` | 0.0612 | 0.0074 |
| 5 | `Geography` | 0.0485 | 0.0041 |
| 6 | `Gender` | 0.0081 | 0.0028 |
| 7 | `Tenure` | 0.0043 | 0.0042 |
| 8 | `CreditScore` | 0.0022 | 0.0050 |
| 9 | `HasCrCard` | 0.0018 | 0.0008 |
| 10 | `EstimatedSalary` | −0.0027 | 0.0016 |

---

## V. Discussion

### A. RQ1: Segment Disparities and Statistical Correction
All four examined dimensions—geography, age band, product count, and active membership—show statistically significant departures from independence after FDR correction. The 46–59 age band has a 51.10% observed churn rate, or 2.51 times the portfolio baseline. Customers with two products have an observed churn rate of 7.58%, compared with 27.71% among single-product customers. These are strong associations within this dataset; they do not show that age, geography, or product ownership causes attrition. In particular, the results do not establish that cross-selling a second product would causally reduce churn.

### B. RQ2: Predictive Validity Under Leakage-Safe Protocols
Under the reported cross-validation and pipeline design, histogram-based gradient boosting outperforms regularized logistic regression by 0.0737 mean AP. On the untouched test set, the selected model achieves AP of 0.6980 and ROC-AUC of 0.8578, exceeding the dummy baseline. This supports retrospective discriminatory ranking within the dataset. Because the dataset has no longitudinal observations or timestamps, the results do not demonstrate prospective forecasting, temporal stability, or performance on future customer cohorts.

### C. RQ3: Thresholds and the Precision–Recall Tension
The 0.50 threshold prioritizes precision and generates 71 false alarms, but misses 214 of 407 churners. The 0.27 threshold improves recall and catches 88 additional churners, but generates 142 additional false alarms. Choosing between them requires financial parameters absent from the dataset, including customer lifetime value, campaign costs, contact capacity, and the relative cost of false positives and false negatives. Thus, this analysis quantifies the trade-off but does not recommend a universal operating threshold.

### D. RQ4: Interpretability and the Causal Boundary
Permutation importance and SHAP identify product count, age, and active membership as prominent sources of predictive information in the fitted model. These rankings describe model reliance under the available features and evaluation design. They do not imply that a change to a feature would change actual customer behavior. For example, fee discounts or product bundling cannot be inferred to prevent churn from these observational associations alone; intervention studies would be needed to assess causal effects.

---

## VI. Threats to Validity and Limitations

1) **Static cross-sectional data:** The constant `Year` field and absence of event timestamps, rolling histories, or repeated customer observations preclude survival forecasting, hazard-rate estimation, and drift detection.
2) **Unverified provenance:** The data resemble an educational churn benchmark, but the provenance of this specific file has not been independently verified. Results should not be interpreted as official European Central Bank data or as representative of specific European commercial banks.
3) **Residual missed churn:** Even at the tuned 0.27 threshold, 30.96% of test-set churners are missed; at 0.50, the miss rate is 52.58%.
4) **No financial cost matrix:** Customer profitability, acquisition costs, retention campaign costs, and intervention outcomes are unavailable. Therefore, a cost-minimizing threshold cannot be determined from the present data.
5) **Non-causal evidence:** Unobserved factors such as service failures, competitor offers, and branch closures may influence both customer characteristics and attrition. The observational analysis cannot isolate their effects.

---

## VII. Conclusion and Future Work

This study provides a leakage-safe empirical benchmark for customer-churn classification using an educational cross-sectional banking dataset. All four segment dimensions survive Benjamini–Hochberg correction, including the elevated churn rate in the 46–59 age band. Histogram-based gradient boosting achieves test AP of 0.6980 and ROC-AUC of 0.8578, outperforming the tested linear baseline. Threshold analysis exposes a meaningful operational trade-off: the default 0.50 cutoff misses 52.58% of churners, while the tuned 0.27 cutoff reduces the miss rate to 30.96% at the expense of more false alarms. Feature attribution identifies variables on which the model relies but does not support causal prescriptions.

Future work should prioritize three directions. First, obtain longitudinal, timestamped banking observations to support temporal validation and survival analysis. Second, incorporate customer lifetime value and intervention costs to select thresholds using an explicit cost-benefit objective. Third, evaluate proactive-retention strategies through prospective randomized trials to determine whether targeted interventions causally affect attrition.

---

## Acknowledgment

The author acknowledges the open-source scientific Python community and the tools used in the analysis, including scikit-learn, SciPy, pandas, NumPy, and SHAP.

---

## References

[1] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems (NeurIPS 30)*, 2017, pp. 4765–4774.  
[2] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge, U.K.: Cambridge University Press, 2009.  
[3] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” *Journal of the Royal Statistical Society: Series B (Methodological)*, vol. 57, no. 1, pp. 289–300, 1995.  
[4] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall/CRC, 1993.  
[5] A. De Caigny, K. Coussement, and K. W. De Bock, “A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees,” *European Journal of Operational Research*, vol. 269, no. 2, pp. 760–772, 2018.  
[6] Y. Xie, X. Li, E. W. T. Ngai, and W. Ying, “Customer churn prediction using improved balanced random forests,” *Expert Systems with Applications*, vol. 36, no. 3, pp. 5445–5449, 2009.  
[7] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.  
[8] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015.  
[9] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1–21, 2012.  
[10] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*, vol. 4, no. 9, Art. no. 100804, 2023.  
[11] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
