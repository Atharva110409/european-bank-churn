# Model Confusion Matrices and Operating Threshold Analysis

This technical report provides a detailed breakdown of classification matrices for all evaluated candidate architectures on the held-out test partition ($n=2,000$, with 407 actual exits and 1,593 retained customers).

---

### 1. Architectural Baseline: DummyClassifier (Strategy="prior")

The naive prior baseline always predicts the majority class (`Retained = 0`).

```
                    PREDICTED RETAINED (0)    PREDICTED EXITED (1)
ACTUAL RETAINED (0)         1,593 (TN)                  0 (FP)
ACTUAL EXITED (1)             407 (FN)                  0 (TP)
```

- **Accuracy:** 79.65% (Illustrates why accuracy is a misleading metric under class imbalance)
- **Precision:** 0.00%
- **Recall:** 0.00% (Misses 100% of all churning customers)
- **$F_1$ Score:** 0.0000
- **Average Precision (AP):** 0.2035 (Matches the positive class base rate)

---

### 2. Selected Model: HistGradientBoosting at Default Threshold (0.50)

Evaluated at the standard probability threshold of $p \ge 0.50$:

```
                    PREDICTED RETAINED (0)    PREDICTED EXITED (1)
ACTUAL RETAINED (0)         1,522 (TN)                 71 (FP)
ACTUAL EXITED (1)             214 (FN)                193 (TP)
```

- **Precision:** **73.11%** ($193 / [193 + 71]$)
- **Recall:** **47.42%** ($193 / [193 + 214]$)
- **Specificity:** **95.54%** ($1,522 / [1,522 + 71]$)
- **$F_1$ Score:** **0.5753**
- **Operational Reality:**
  - High precision: nearly 3 out of 4 flagged accounts actually churn.
  - Severe recall limitation: **214 churning customers (52.58% of all exits) are completely missed** by the model at this threshold.

---

### 3. Selected Model: HistGradientBoosting at CV-Tuned Alternative Threshold (0.27)

Threshold tuned on training cross-validation to maximize the $F_1$ score ($p \ge 0.27$):

```
                    PREDICTED RETAINED (0)    PREDICTED EXITED (1)
ACTUAL RETAINED (0)         1,380 (TN)                213 (FP)
ACTUAL EXITED (1)             126 (FN)                281 (TP)
```

- **Precision:** **56.88%** ($281 / [281 + 213]$)
- **Recall:** **69.04%** ($281 / [281 + 126]$)
- **Specificity:** **86.63%** ($1,380 / [1,380 + 213]$)
- **$F_1$ Score:** **0.6238**
- **Operational Reality:**
  - Detects **88 additional churners** (increasing recall from 47.42% to 69.04%).
  - Increases false positive contacts from 71 to 213 (+142 loyal customers flagged).

---

### 4. Side-by-Side Trade-Off Summary

| Diagnostic Metric | Baseline (Dummy) | Default Threshold (0.50) | CV-Selected Alternative (0.27) |
| :--- | :--- | :--- | :--- |
| **True Positives (Exits Caught)** | 0 | 193 | 281 |
| **False Negatives (Exits Missed)** | 407 | 214 | 126 |
| **False Positives (False Alarms)** | 0 | 71 | 213 |
| **True Negatives (Correct Stay)** | 1,593 | 1,522 | 1,380 |
| **Precision** | 0.00% | **73.11%** | **56.88%** |
| **Recall** | 0.00% | **47.42%** | **69.04%** |
| **$F_1$ Score** | 0.0000 | **0.5753** | **0.6238** |
| **Miss Rate** | 100.00% | **52.58%** | **30.96%** |

### 5. Business Decision Framework

The choice between these two operating points cannot be solved by machine learning alone:
1. **Choose 0.50 if:** The cost of retention outreach is high, intervention capacity is severely constrained, or contacting customers carries high customer friction/annoyance risk.
2. **Choose 0.27 if:** The cost of losing a customer vastly outweighs the cost of marketing outreach, and automated digital communications (e.g. low-cost in-app retention offers) are deployed.
