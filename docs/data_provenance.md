# Dataset Provenance, Scope, and Data Limitations

---

### 1. Verbatim Provenance Statement

> The file resembles the widely circulated educational `Churn_Modelling` benchmark, but the exact provenance of this specific CSV has not been independently verified. Describe it as an educational customer-churn dataset supplied for this project. Do not present it as actual European Central Bank data or proprietary records from a European bank.

---

### 2. Context & Background

The dataset `European_Bank.csv` supplied for this project shares structural and statistical identicality with the standard synthetic educational dataset frequently circulated in academic machine-learning tutorials under the title `Churn_Modelling.csv`. 

While the records provide a realistic tabular distribution for practicing data validation, cohort segmentation, and leak-free machine-learning pipelines, there is no verified chain of custody connecting these records to the European Central Bank (ECB) or any regulated commercial credit institution within the Eurozone.

Presenting these figures to stakeholders, regulators, or corporate leadership as empirical evidence of real European banking behavior would be factually inaccurate and methodologically irresponsible.

---

### 3. What the Data Can and Cannot Support

To protect analytical integrity, the boundaries of what this dataset supports are strictly defined below:

| Dimension | Supported by Current Data | NOT Supported Without Additional Data |
| :--- | :--- | :--- |
| **Geography** | Descriptive differences across the 3 sampled countries in this file (France, Germany, Spain). | Macroeconomic claims regarding European banking stability, cross-border capital flight, or regulatory differences across Eurozone member states. |
| **Demographics & Behavior** | Statistical associations between customer attributes (e.g., age, product holdings, balance) and historical churn status. | Proof or causal evidence that age, inactivity, or balance levels *cause* customers to terminate their banking relationships. |
| **Classification Objective** | Retrospective scoring: estimating whether an archived record is associated with the historical label `Exited = 1`. | Prospective forecasting: predicting *when* a customer will leave, hazard rate over time, or whether a specific retention discount would prevent departure. |
| **Financial Analysis** | Using customer account balance as an ordinal indicator of financial activity profile. | Calculating actual dollar loss, customer lifetime value (LTV), revenue at risk, net interest margin impact, or return on investment (ROI) for retention marketing. |

---

### 4. Structural Data Characteristics

- **Sample Size:** Exactly 10,000 unique records.
- **Completeness:** 0 missing values across all 14 columns.
- **Uniqueness:** 0 duplicate rows; 10,000 unique `CustomerId` values.
- **Target Distribution:** 2,037 exited customers (20.37%), 7,963 retained customers (79.63%).
- **Time Field:** The single `Year` column contains the constant value `2025` for every record. There are no monthly snapshots, no transaction timestamps, and no cohort tenure dates. Consequently, no temporal time-series, longitudinal panel regressions, or survival analysis can be conducted on this file.
