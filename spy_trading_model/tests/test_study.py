import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np

from spy_signals import Config, explain, report, study as S
from spy_signals.synthetic import make_synthetic_ohlcv

OH = make_synthetic_ohlcv("2010-01-01", "2025-12-31", 3)


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = S.train_all(Config(fast=True), OH)

    def test_all_models_and_baselines_present_for_both_stages(self):
        self.assertEqual(set(self.st.runs), {"baseline_majority", "baseline_stratified_random", "logreg", "random_forest", "grad_boosting"})
        for stage in ("validation", "test"):
            tbl = S.classification_table(self.st, stage)
            self.assertEqual(len(tbl), 5)
            self.assertTrue(tbl["macro_f1"].between(0, 1).all())
            bt = S.backtest_stage(self.st, stage, 0.001, 0.0005, engine="builtin")
            self.assertEqual(len(bt["perf"]), 6)           # 5 strategies + Buy & Hold

    def test_fee_slider_range_is_monotone_for_traders(self):
        lo = S.backtest_stage(self.st, "test", 0.0, 0.0, engine="builtin")["perf"]
        hi = S.backtest_stage(self.st, "test", 0.02, 0.0, engine="builtin")["perf"]
        for n in ("logreg", "random_forest", "grad_boosting", "baseline_stratified_random"):
            if lo.loc[n, "n_orders"] > 0:
                self.assertLess(hi.loc[n, "final_value"], lo.loc[n, "final_value"])
        self.assertEqual(lo.loc["baseline_majority", "n_orders"] > 0, hi.loc["baseline_majority", "n_orders"] > 0)

    def test_rules_and_confidence_filter(self):
        a = S.backtest_stage(self.st, "test", 0.001, 0.0005, "stateful", 0.0, "builtin")["perf"]
        b = S.backtest_stage(self.st, "test", 0.001, 0.0005, "daily", 0.0, "builtin")["perf"]
        c = S.backtest_stage(self.st, "test", 0.001, 0.0005, "stateful", 0.9, "builtin")["perf"]
        self.assertGreaterEqual(b.loc["random_forest", "n_orders"], a.loc["random_forest", "n_orders"] - 2)
        self.assertLessEqual(c.loc["random_forest", "n_orders"], a.loc["random_forest", "n_orders"])

    def test_sweep_report_and_permutation(self):
        fees = [0.0, 0.01, 0.02]
        sw = S.fee_sweep(self.st, "test", ["logreg"], fees, 0.0005, engine="builtin")
        self.assertEqual(set(sw["fee"]), set(fees))
        res = S.report_inputs(self.st, 0.02, 0.0005, "builtin")
        md = report.build_report(res["cfg"], res)
        self.assertNotIn("nan%", md)
        self.assertIn("Lessons Learned", md)
        self.assertEqual(len(S.permutation_for(self.st, "logreg")), self.st.ds.X.shape[1])

    def test_extreme_thresholds(self):
        with self.assertRaises(S.StudyError):
            S.train_all(Config(fast=True, up_threshold=0.2, down_threshold=0.2), OH)
        st = S.train_all(Config(fast=True, up_threshold=0.01, down_threshold=0.2), OH)   # Sell absent -> binary
        coefs = explain.logreg_coefficients(st.runs["logreg"].train_only_model, st.ds.X.columns)
        self.assertEqual(coefs.shape[1], 2)
        tbl = explain.explain_prediction_logreg(st.runs["logreg"].train_only_model, st.splits.X_val.iloc[[10]])
        self.assertTrue(np.isfinite(tbl["contribution"]).all())
        self.assertTrue((st.runs["logreg"].test_pred["Sell"] == 0).all())


if __name__ == "__main__":
    unittest.main()
