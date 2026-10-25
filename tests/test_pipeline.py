"""
test_pipeline.py
Quality gate unit and smoke tests.

Verifies:
1. Feature policy and target leakage prevention (Exited never in X).
2. Segment group counts summing to the full dataset (10,000 customers).
3. Graceful handling of zero-result filter states without exceptions.
4. Artifact file presence, schema integrity, and exact AP labeling.
5. Model consistency (model.joblib reproduces test predictions).
6. Streamlit application scripts startup and import verification.
"""

import os
import json
import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

from src.data_validation import load_and_validate_data, get_features_and_target
from src.segmentation import (
    summarize_geography,
    summarize_age_bands,
    summarize_num_of_products,
    summarize_is_active_member,
    get_all_segment_summaries
)


class TestEuropeanBankChurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_csv = "data/raw/European_Bank.csv"
        cls.df, cls.audit = load_and_validate_data(cls.raw_csv)
        cls.total_rows = 10000
        cls.total_exited = 2037

    def test_feature_policy_no_leakage(self):
        """Quality Gate: Exited must never be in X; CustomerId, Surname, Year excluded."""
        X, y = get_features_and_target(self.df)
        self.assertNotIn("Exited", X.columns, "Leakage: Exited found in feature matrix X!")
        self.assertNotIn("CustomerId", X.columns, "CustomerId found in feature matrix X!")
        self.assertNotIn("Surname", X.columns, "Surname found in feature matrix X!")
        self.assertNotIn("Year", X.columns, "Year found in feature matrix X!")
        self.assertEqual(len(X), self.total_rows)
        self.assertEqual(len(y), self.total_rows)
        self.assertEqual(X.shape[1], 10)

    def test_segment_group_counts_sum_to_full_dataset(self):
        """Quality Gate: All segment groups must sum to 10,000 customers and 2,037 exits."""
        summaries = get_all_segment_summaries(self.df, total_exits=self.total_exited)
        
        for name, table in summaries.items():
            sum_n = int(table["n"].sum())
            sum_exited = int(table["exited"].sum())
            sum_retained = int(table["retained"].sum())
            
            self.assertEqual(sum_n, self.total_rows, f"{name} n does not sum to {self.total_rows} (got {sum_n})")
            self.assertEqual(sum_exited, self.total_exited, f"{name} exited does not sum to {self.total_exited} (got {sum_exited})")
            self.assertEqual(sum_retained, self.total_rows - self.total_exited, f"{name} retained mismatch")
            
            # Check share_of_all_exits sum is approximately 1.0
            sum_share = float(table["share_of_all_exits"].sum())
            self.assertAlmostEqual(sum_share, 1.0, places=4, msg=f"{name} share of exits does not sum to 1.0")

    def test_zero_result_filter_handling(self):
        """Quality Gate: Empty filtered dataframe must not crash segmentation functions."""
        empty_df = self.df.iloc[0:0]
        
        geo = summarize_geography(empty_df, total_exits=self.total_exited)
        self.assertEqual(int(geo["n"].sum()), 0)
        self.assertEqual(int(geo["exited"].sum()), 0)
        self.assertTrue((geo["churn_rate"] == 0.0).all())
        
        age = summarize_age_bands(empty_df, total_exits=self.total_exited)
        self.assertEqual(int(age["n"].sum()), 0)
        self.assertEqual(int(age["exited"].sum()), 0)
        self.assertTrue((age["churn_rate"] == 0.0).all())
        
        prod = summarize_num_of_products(empty_df, total_exits=self.total_exited)
        self.assertEqual(int(prod["n"].sum()), 0)
        self.assertEqual(int(prod["exited"].sum()), 0)
        self.assertTrue((prod["churn_rate"] == 0.0).all())
        
        act = summarize_is_active_member(empty_df, total_exits=self.total_exited)
        self.assertEqual(int(act["n"].sum()), 0)
        self.assertEqual(int(act["exited"].sum()), 0)
        self.assertTrue((act["churn_rate"] == 0.0).all())

    def test_artifact_presence_and_schema(self):
        """Quality Gate: Required artifacts must exist, have valid structure, and proper labeling."""
        model_path = "artifacts/model.joblib"
        metrics_path = "artifacts/metrics.json"
        preds_path = "artifacts/test_predictions.csv"
        
        self.assertTrue(os.path.exists(model_path), f"Missing {model_path}")
        self.assertTrue(os.path.exists(metrics_path), f"Missing {metrics_path}")
        self.assertTrue(os.path.exists(preds_path), f"Missing {preds_path}")
        
        # Verify metrics.json
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        self.assertIn("model_selection", metrics)
        self.assertIn("cv_comparison", metrics)
        self.assertIn("held_out_test_evaluation", metrics)
        
        # Assert Average Precision is labeled "AP", not generically "PR-AUC"
        self.assertIn("AP", metrics["held_out_test_evaluation"])
        self.assertNotIn("PR-AUC", metrics["held_out_test_evaluation"])
        self.assertEqual(metrics["model_selection"]["metric"], "Average Precision (AP)")
        
        # Verify test_predictions.csv
        preds = pd.read_csv(preds_path)
        self.assertEqual(list(preds.columns), ["y_true", "predicted_probability", "predicted_label"])
        self.assertEqual(len(preds), 2000, "Held-out test set size must be exactly 2,000 (20% of 10,000)")
        self.assertNotIn("CustomerId", preds.columns)
        self.assertNotIn("Surname", preds.columns)
        
        # Verify model.joblib can load and run
        loaded_model = joblib.load(model_path)
        self.assertTrue(hasattr(loaded_model, "predict_proba"))

    def test_saved_model_exact_reproducibility(self):
        """Quality Gate: artifacts/model.joblib must reproduce identical probabilities on test split."""
        from sklearn.model_selection import train_test_split
        X, y = get_features_and_target(self.df)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        loaded_model = joblib.load("artifacts/model.joblib")
        preds = pd.read_csv("artifacts/test_predictions.csv")
        
        reproduced_probs = loaded_model.predict_proba(X_test)[:, 1]
        np.testing.assert_allclose(
            reproduced_probs,
            preds["predicted_probability"].values,
            rtol=1e-5,
            err_msg="Model predictions diverged: saved pipeline is not identical!"
        )

    def test_app_pages_startup(self):
        """Quality Gate: All Streamlit pages must initialize and run without exceptions."""
        from streamlit.testing.v1 import AppTest
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for page_path in ["app.py", "pages/1_Segments.py", "pages/2_Model_Performance.py", "pages/3_Explainability.py"]:
            abs_page = os.path.join(project_root, page_path)
            at = AppTest.from_file(abs_page)
            at.run(timeout=10)
            self.assertFalse(at.exception, f"Streamlit page {page_path} crashed with: {at.exception}")

    def test_explainability_artifacts_and_properties(self):
        """Quality Gate: SHAP plot and permutation importance artifacts must exist and satisfy schema/range properties."""
        shap_plot_path = "reports/shap_summary_global.png"
        perm_csv_path = "reports/permutation_importance.csv"
        
        self.assertTrue(os.path.exists(shap_plot_path), f"Missing {shap_plot_path}")
        self.assertGreater(os.path.getsize(shap_plot_path), 1000, "SHAP plot image is empty or incomplete")
        
        self.assertTrue(os.path.exists(perm_csv_path), f"Missing {perm_csv_path}")
        perm_df = pd.read_csv(perm_csv_path)
        
        # Verify columns and feature count
        self.assertIn("feature", perm_df.columns)
        self.assertIn("predictive_importance_mean", perm_df.columns)
        self.assertEqual(len(perm_df), 10, "Permutation importance must include all 10 candidate features")
        
        # Verify predictive importance values are non-negative for primary predictive features
        top_importance = perm_df.iloc[0]["predictive_importance_mean"]
        self.assertGreater(top_importance, 0.0, "Top feature importance must be positive")
        
        # Verify explicit disclaimer is attached to records
        self.assertTrue(
            (perm_df["interpretability_notice"] == "Predictive importance, not causal importance").all(),
            "Permutation importance table must explicitly label predictive rather than causal importance"
        )

    def test_statistical_audit_properties(self):
        """Quality Gate: Statistical audit must verify 4 chi-square tests, FDR correction, and valid bootstrap CIs."""
        from src.statistics import benjamini_hochberg_correction
        
        # 1. Verify Benjamini-Hochberg monotonic ordering logic
        dummy_p = [0.04, 0.01, 0.03, 0.02]
        adj_p, sig = benjamini_hochberg_correction(dummy_p, q=0.05)
        for r_p, a_p in zip(dummy_p, adj_p):
            self.assertGreaterEqual(a_p, r_p, f"FDR adjusted p {a_p} cannot be less than raw p {r_p}")
            self.assertLessEqual(a_p, 1.0)
            
        # 2. Check saved segment_significance_tests.csv
        sig_csv_path = "reports/segment_significance_tests.csv"
        self.assertTrue(os.path.exists(sig_csv_path), f"Missing {sig_csv_path}")
        sig_df = pd.read_csv(sig_csv_path)
        self.assertEqual(len(sig_df), 4, "Must contain exactly 4 segmentation dimensions")
        
        expected_dims = {"Geography", "AgeBand", "NumOfProducts", "IsActiveMember"}
        self.assertEqual(set(sig_df["dimension"]), expected_dims)
        
        for _, row in sig_df.iterrows():
            raw_p = float(row["raw_p_value"])
            adj_p = float(row["fdr_adjusted_p_value"])
            self.assertGreaterEqual(raw_p, 0.0)
            self.assertLessEqual(raw_p, 1.0)
            self.assertGreaterEqual(adj_p, 0.0)
            self.assertLessEqual(adj_p, 1.0)
            self.assertGreaterEqual(adj_p, raw_p, f"Adjusted p {adj_p} < raw p {raw_p} for {row['dimension']}")
            self.assertGreater(float(row["chi2_statistic"]), 0.0)
            self.assertGreater(float(row["cramers_v"]), 0.0)
            self.assertTrue(row["significant_fdr_05"])

        # 3. Check saved bootstrap_cis.json
        boot_json_path = "reports/bootstrap_cis.json"
        self.assertTrue(os.path.exists(boot_json_path), f"Missing {boot_json_path}")
        with open(boot_json_path, "r", encoding="utf-8") as f:
            boot_data = json.load(f)
            
        required_metrics = [
            "overall_churn_rate",
            "age_46_59_churn_rate",
            "test_average_precision_ap",
            "test_roc_auc"
        ]
        for m in required_metrics:
            self.assertIn(m, boot_data, f"Metric {m} missing from bootstrap CIs")
            pt = boot_data[m]["point_estimate"]
            lower = boot_data[m]["ci_lower_95"]
            upper = boot_data[m]["ci_upper_95"]
            self.assertLess(lower, pt, f"CI lower bound {lower} not strictly less than point estimate {pt} for {m}")
            self.assertLess(pt, upper, f"Point estimate {pt} not strictly less than CI upper bound {upper} for {m}")


if __name__ == "__main__":
    unittest.main()


