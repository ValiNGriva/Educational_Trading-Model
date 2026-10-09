"""Run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from spy_signals import Config, audit, backtest
from spy_signals.features import FEATURE_NAMES, build_features
from spy_signals.pipeline import prepare_data, run_experiment
from spy_signals.splits import chronological_split
from spy_signals.synthetic import make_synthetic_ohlcv
from spy_signals.target import make_target

OHLCV = make_synthetic_ohlcv("2010-01-01", "2025-12-31", 7)


class TestFeatures(unittest.TestCase):
    def test_columns_and_no_raw_price_levels(self):
        f = build_features(OHLCV)
        self.assertEqual(list(f.columns), FEATURE_NAMES)
        self.assertTrue((f.dropna().abs().max() < 1e4).all())  # nothing price-scaled

    def test_truncation_audit_passes(self):
        self.assertTrue(audit.audit_no_lookahead(OHLCV)[1])

    def test_truncation_audit_catches_planted_leak(self):
        """If a feature secretly used tomorrow's close, the audit must FAIL."""
        real = audit.build_features

        def leaky(df):
            f = real(df).copy()
            f["leak"] = df["Close"].shift(-1) / df["Close"] - 1
            return f

        audit.build_features = leaky
        try:
            self.assertFalse(audit.audit_no_lookahead(OHLCV)[1])
        finally:
            audit.build_features = real


class TestTarget(unittest.TestCase):
    def test_labels_and_nan_tail(self):
        close = pd.Series([100, 100, 100, 100, 100, 102, 98, 100.5, 100, 100, 100, 100.0])
        t = make_target(close, horizon=5, up=0.01, down=0.01)
        self.assertEqual(t["label"].iloc[0], 1)       # 100 -> 102
        self.assertEqual(t["label"].iloc[1], -1)      # 100 -> 98
        self.assertEqual(t["label"].iloc[2], 0)       # 100 -> 100.5
        self.assertTrue(t["label"].iloc[-5:].isna().all())


class TestSplits(unittest.TestCase):
    def test_order_gap_and_audit(self):
        cfg = Config(fast=True)
        ds = prepare_data(cfg, OHLCV)
        sp = chronological_split(ds.X, ds.y, cfg)
        self.assertLess(sp.X_train.index.max(), sp.X_val.index.min())
        self.assertLess(sp.X_val.index.max(), sp.X_test.index.min())
        self.assertEqual(sp.purged_train, cfg.horizon)
        self.assertTrue(audit.audit_split_order(ds.X, sp, OHLCV.index, cfg.horizon)[1])

    def test_no_label_horizon_overlaps_next_split(self):
        cfg = Config(fast=True)
        ds = prepare_data(cfg, OHLCV)
        sp = chronological_split(ds.X, ds.y, cfg)
        pos = OHLCV.index.get_loc
        self.assertLess(pos(sp.X_train.index.max()) + cfg.horizon, pos(sp.X_val.index.min()))


class TestBacktestEngine(unittest.TestCase):
    def _px(self):
        idx = pd.bdate_range("2024-01-01", periods=6)
        return pd.DataFrame({"Open": [100, 100, 110, 120, 130, 130.0], "High": 200.0, "Low": 1.0,
                             "Close": [100, 105, 115, 125, 130, 130.0], "Volume": 1}, index=idx)

    def test_zero_cost_round_trip(self):
        px = self._px()
        desired = pd.Series([0, 1, 1, 0, 0, 0], index=px.index)  # decide long at close d1, flat at close d3
        r = backtest.run_backtest(px, desired, "t", 0.0, 0.0, 1000.0, engine="builtin")
        # buy at open d2 (110), sell at open d4 (130)  -> 1000 * 130/110
        self.assertAlmostEqual(r.equity.iloc[-1], 1000 * 130 / 110, places=6)
        self.assertEqual(r.n_orders, 2)

    def test_costs_reduce_value_exactly(self):
        px = self._px()
        desired = pd.Series([0, 1, 1, 0, 0, 0], index=px.index)
        fee, slip = 0.001, 0.0005
        r = backtest.run_backtest(px, desired, "t", fee, slip, 1000.0, engine="builtin")
        expected = 1000 * (130 * (1 - slip) * (1 - fee)) / (110 * (1 + slip) * (1 + fee))
        self.assertAlmostEqual(r.equity.iloc[-1], expected, places=6)

    def test_signal_is_filled_next_open_not_same_day(self):
        px = self._px()
        desired = pd.Series([1, 1, 1, 1, 1, 1], index=px.index)
        r = backtest.run_backtest(px, desired, "t", 0.0, 0.0, 1000.0, engine="builtin")
        # day-0 close signal -> first fill is day-1 open (100), NOT day-0
        self.assertAlmostEqual(r.equity.iloc[0], 1000.0)
        self.assertAlmostEqual(r.equity.iloc[-1], 1000 * 130 / 100, places=6)

    def test_stateful_vs_daily_positions(self):
        pred = pd.Series([1, 0, 0, -1, 0, 1, 1, 0])
        self.assertEqual(backtest.signals_to_positions(pred, mode="stateful").tolist(), [1, 1, 1, 0, 0, 1, 1, 1])
        self.assertEqual(backtest.signals_to_positions(pred, mode="daily").tolist(), [1, 0, 0, 0, 0, 1, 1, 0])

    def test_more_trading_costs_more(self):
        px = OHLCV.loc["2023-01-01":"2023-12-31"]
        flip = pd.Series(np.tile([1, 0], len(px))[: len(px)], index=px.index)
        a = backtest.run_backtest(px, flip, "f", 0.0, 0.0, 1e5, engine="builtin").equity.iloc[-1]
        b = backtest.run_backtest(px, flip, "f", 0.001, 0.0005, 1e5, engine="builtin").equity.iloc[-1]
        self.assertLess(b, a)


class TestEndToEnd(unittest.TestCase):
    def test_pipeline_runs_and_audits_pass(self):
        import tempfile
        cfg = Config(fast=True, engine="builtin", output_dir=tempfile.mkdtemp())
        res = run_experiment(cfg, OHLCV, verbose=False)
        self.assertTrue((res["audit"]["passed"] == "PASS").all(), res["audit"])
        self.assertIn("Buy & Hold SPY", res["perf"].index)
        self.assertGreater(res["buy_hold"].equity.iloc[-1], 0)
        # same window for strategy and benchmark
        self.assertTrue(res["buy_hold"].equity.index.equals(res["results"]["Model (stateful)"].equity.index))


if __name__ == "__main__":
    unittest.main()
