# Executive Summary: Customer Churn Analytics MVP

**Target Audience:** Executive Leadership, Retail Banking Operations, and Strategy  
**Source Dataset:** Educational Customer Churn Benchmark (`European_Bank.csv`, $N=10,000$)

---

### Headline Strategic Finding
Retail banking customer churn in this portfolio is heavily concentrated in a single demographic cohort: **customers aged 46 to 59 exhibit an observed churn rate of 51.10%** (810 exits out of 1,585 customers). This is **2.51 times the overall portfolio churn rate of 20.37%**. While this age bracket represents only 15.85% of total accounts, it accounts for **nearly 40% of all customer exits**. In addition, accounts holding exactly two products exhibit strong loyalty (only 7.58% churn), whereas single-product accounts churn at 27.71%.

---

### What the Machine Learning Model Does
We developed and benchmarked a retrospective risk-scoring classifier using a leak-free machine learning workflow. Evaluated strictly on cross-validation and a 2,000-customer held-out test split, the winning algorithm (**Histogram-based Gradient Boosting**) achieved an **Average Precision of 0.6980** (versus 0.2035 for a naive baseline) and a **ROC-AUC of 0.8578**, decisively outperforming standard linear models.

---

### The Operating Trade-Off: A Business Decision, Not a Solved Algorithm
Machine learning does not decide how aggressive a bank should be in contacting customers. The choice of decision threshold represents a fundamental operational trade-off:

- **At the Default Threshold (0.50):** The model prioritizes precision. When it flags a customer, it is right nearly **73% of the time (73.1% precision)** with only 71 false alarms. However, it catches only **193 of 407 actual churners (47.4% recall)**, meaning it **misses over half of all departures (52.6% missed)**.
- **At the Tuned Alternative Threshold (0.27):** The model prioritizes detection. It catches **281 of 407 churners (69.0% recall)**, but false alarms increase to **213** (precision drops to 56.9%, with 31.0% missed).

*Strategic Reality:* Choosing between these operating points depends entirely on institutional unit economics: does the financial loss of an unaddressed customer exit outweigh the marketing and operational expense of contacting loyal customers? Because this benchmark dataset lacks cost and revenue metrics, this MVP reports both thresholds rather than prescribing an arbitrary business rule.

---

### Concrete Next-Step Recommendations

1. **Investigate Age 46–59 Value Proposition:** Conduct focused qualitative customer discovery and product reviews to identify why customers aged 46–59 exit at more than double the bank average. Evaluate competitive wealth-management or fee-structure alternatives for this cohort.
2. **Promote Multi-Product Deepening:** Because dual-product accounts experience by far the lowest churn rate (7.58%), design onboarding programs that incentivize single-product customers (27.71% churn) to adopt a secondary service within their first six months.
3. **Collect Longitudinal and Unit-Cost Data:** Transition from static snapshot analytics to longitudinal tracking. Recording monthly balance flows, transaction frequency, and retention marketing costs will enable prospective time-series forecasting and precise ROI modeling.

---

### Key Limitations in Plain English
- **Not a Forecasting Tool:** This system scores historical customer records from a single point in time. It cannot predict *when* an account will leave or how churn risk evolves over future months.
- **Correlations, Not Causes:** Statistical links between churn, age, and product counts are associations, not proof that age causes churn. Retention interventions must be tested experimentally before broad rollout.
- **No Revenue or Cost Data:** The dataset contains no profit margin, lifetime value (LTV), or campaign cost figures; no financial savings or ROI claims should be derived from this deliverable.
- **Unverified Provenance:** This is an educational benchmark dataset and must not be presented as real European Central Bank or commercial bank data.
