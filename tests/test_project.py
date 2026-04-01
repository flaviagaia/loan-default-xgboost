from __future__ import annotations

from pathlib import Path
import unittest

from src.modeling import run_project


class LoanDefaultXGBoostTestCase(unittest.TestCase):
    def test_report_has_expected_shape(self) -> None:
        report = run_project(Path(__file__).resolve().parents[1])
        self.assertEqual(report["project_name"], "loan_default_xgboost")
        self.assertGreater(report["dataset_rows"], 1000)
        self.assertIn("roc_auc", report["selected_model_metrics"])
        self.assertGreater(report["selected_model_metrics"]["roc_auc"], 0.8)
        self.assertEqual(len(report["top_feature_importance"]), 8)


if __name__ == "__main__":
    unittest.main()

