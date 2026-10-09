#!/usr/bin/env python
"""One command, whole project:  python run_pipeline.py            (real SPY via yfinance)
                                python run_pipeline.py --source synthetic --fast   (offline smoke test)
Outputs (figures, tables, metrics.json, lessons_learned.md) go to ./outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from spy_signals import Config, plotting as P  # noqa: E402
from spy_signals.data import load_ohlcv  # noqa: E402
from spy_signals.explain import logreg_coefficients  # noqa: E402
from spy_signals.pipeline import run_experiment  # noqa: E402
from spy_signals.report import build_report  # noqa: E402
from spy_signals.target import make_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="auto | yfinance | csv | synthetic")
    ap.add_argument("--fast", action="store_true", help="small hyper-parameter grids")
    ap.add_argument("--engine", default="auto", choices=["auto", "vectorbt", "builtin"])
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--train-end"); ap.add_argument("--val-end")
    ap.add_argument("--horizon", type=int); ap.add_argument("--up", type=float); ap.add_argument("--down", type=float)
    ap.add_argument("--fee", type=float); ap.add_argument("--slippage", type=float)
    ap.add_argument("--min-confidence", type=float)
    ap.add_argument("--output-dir")
    a = ap.parse_args()

    cfg = Config(fast=a.fast, engine=a.engine)
    for arg, field in (("start", "start"), ("end", "end"), ("train_end", "train_end"), ("val_end", "val_end"),
                       ("horizon", "horizon"), ("up", "up_threshold"), ("down", "down_threshold"), ("fee", "fee"),
                       ("slippage", "slippage"), ("min_confidence", "min_confidence"), ("output_dir", "output_dir")):
        if getattr(a, arg) is not None:
            setattr(cfg, field, getattr(a, arg))

    ohlcv = load_ohlcv(cfg, a.source)
    note = ("SYNTHETIC DATA - code-path smoke test only; these numbers say nothing about the real market."
            if (a.source or "").lower() == "synthetic" else "")
    res = run_experiment(cfg, ohlcv)

    fig = Path(cfg.output_dir) / "figures"
    ds, sp = res["ds"], res["splits"]
    P.plot_price_with_splits(ohlcv, cfg, fig / "01_price_splits.png")
    P.plot_class_distribution({"train": sp.y_train, "validation": sp.y_val, "test": sp.y_test}, fig / "02_class_distribution.png")
    P.plot_feature_distributions(ds.X, fig / "03_feature_distributions.png")
    P.plot_correlation(ds.X, fig / "04_correlation.png")
    P.plot_confusion(res["test_summary"]["confusion"], f"Test confusion - {res['best_name']}", fig / "05_confusion_test.png")
    P.plot_importance(res["perm"], fig / "06_permutation_importance.png")
    if res["best_name"] == "logreg":
        P.plot_logreg_coefs(logreg_coefficients(res["train_only"], ds.X.columns), fig / "07_logreg_coefficients.png")
    P.plot_equity([res["buy_hold"], *res["results"].values()], fig / "08_equity_vs_buy_hold.png")
    P.plot_fee_sensitivity(res["sens"], fig / "09_fee_sensitivity.png")

    report = build_report(cfg, res, note)
    (Path(cfg.output_dir) / "lessons_learned.md").write_text(report)
    print("\n" + res["perf"].round(4).to_string())
    print("\nAudit:\n" + res["audit"].to_string(index=False))
    print(f"\nTest accuracy {res['test_summary']['accuracy']:.3f}  |  outputs in {cfg.output_dir}")


if __name__ == "__main__":
    main()
