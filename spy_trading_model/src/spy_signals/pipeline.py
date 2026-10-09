"""End-to-end experiment. The notebooks call these same building blocks step by step."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import audit, backtest, evaluate, explain, models
from .config import Config
from .features import build_features
from .splits import Splits, chronological_split
from .target import make_target


@dataclass
class Dataset:
    X_all: pd.DataFrame      # every date with a complete feature row (incl. last h days with no label yet)
    X: pd.DataFrame          # rows that also have a label
    y: pd.Series
    fwd_ret: pd.Series


def prepare_data(cfg: Config, ohlcv: pd.DataFrame) -> Dataset:
    feats = build_features(ohlcv)
    tgt = make_target(ohlcv["Close"], cfg.horizon, cfg.up_threshold, cfg.down_threshold)
    X_all = feats.dropna()
    both = X_all.join(tgt).dropna(subset=["label"])
    return Dataset(X_all, both[feats.columns], both["label"].astype(int), both["fwd_ret"])


def fit_and_compare(cfg: Config, splits: Splits, verbose: bool = True):
    """Tune each model with purged TimeSeriesSplit on TRAIN, then score on VALIDATION."""
    fitted, rows, cv_scores = {}, {}, {}
    for name, est in models.baseline_models().items():
        est.fit(splits.X_train, splits.y_train)
        rows[name] = evaluate.classification_summary(splits.y_val, est.predict(splits.X_val))
    for name, (est, grid) in models.model_zoo(cfg).items():
        search = models.tune(name, est, grid, splits.X_train, splits.y_train, cfg)
        fitted[name] = search
        cv_scores[name] = {"best_cv_" + cfg.scoring: float(search.best_score_), "best_params": search.best_params_}
        rows[name] = evaluate.classification_summary(splits.y_val, search.predict(splits.X_val))
        if verbose:
            print(f"[{name}] CV {cfg.scoring}={search.best_score_:.4f}  best={search.best_params_}")
    return fitted, rows, cv_scores


def run_experiment(cfg: Config, ohlcv: pd.DataFrame, verbose: bool = True) -> dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. data -> features -> target -> chronological split
    ds = prepare_data(cfg, ohlcv)
    splits = chronological_split(ds.X, ds.y, cfg)
    if verbose:
        print(splits.summary().to_string(index=False))

    # 3. tune on train, compare on validation, choose by validation macro-F1
    fitted, val_rows, cv_scores = fit_and_compare(cfg, splits, verbose)
    real = [m for m in fitted]
    best_name = max(real, key=lambda m: val_rows[m]["macro_f1"])
    if verbose:
        print(f"\nSelected on validation: {best_name}")

    # 4. explainability on VALIDATION with the train-only model (test stays untouched)
    train_only = fitted[best_name].best_estimator_
    perm = explain.permutation_table(train_only, splits.X_val, splits.y_val, cfg.scoring, seed=cfg.random_state)

    # 5. refit on train+val (purged) -> evaluate on test exactly once
    from sklearn.base import clone
    final = clone(fitted[best_name].best_estimator_).fit(splits.X_trainval, splits.y_trainval)
    test_pred = final.predict(splits.X_test)
    test_summary = evaluate.classification_summary(splits.y_test, test_pred)
    test_summary["majority_baseline_accuracy"] = evaluate.majority_accuracy(splits.y_trainval, splits.y_test)

    # 6. live signals for EVERY test-period date (incl. last h days that have no label yet)
    X_live = ds.X_all.loc[ds.X_all.index > pd.Timestamp(cfg.val_end)]
    proba = pd.DataFrame(final.predict_proba(X_live), index=X_live.index,
                         columns=[evaluate.NAMES[c] for c in final.named_steps["clf"].classes_])
    pred = pd.Series(final.predict(X_live), index=X_live.index, name="pred")
    conf = proba.max(axis=1)
    predictions = proba.assign(pred=pred, confidence=conf)
    predictions.to_csv(out_dir / "test_predictions.csv")
    joblib.dump(final, out_dir / "final_model.joblib")

    # 7. backtests with fees + slippage vs buy & hold
    desired = {
        "Model (stateful)": backtest.signals_to_positions(pred, conf, cfg.min_confidence, "stateful"),
        "Model (daily Buy-only)": backtest.signals_to_positions(pred, conf, cfg.min_confidence, "daily"),
    }
    results = {n: backtest.run_backtest(ohlcv, p, n, cfg.fee, cfg.slippage, cfg.init_cash, cfg.engine)
               for n, p in desired.items()}
    bh = backtest.buy_and_hold(ohlcv, X_live.index, cfg.fee, cfg.slippage, cfg.init_cash, cfg.engine)
    perf = pd.DataFrame([bh.metrics()] + [r.metrics() for r in results.values()]).set_index("strategy")
    sens = backtest.fee_sensitivity(ohlcv, desired, [0.0, 0.0005, 0.001, 0.002, 0.005], cfg.slippage,
                                    cfg.init_cash, cfg.engine)

    # 8. leakage audits
    cv = models.time_series_cv(cfg)
    audits = [
        audit.audit_no_lookahead(ohlcv),
        audit.audit_target_not_in_features(ds.X, ds.fwd_ret),
        audit.audit_split_order(ds.X, splits, ohlcv.index, cfg.horizon),
        audit.audit_no_shuffle_cv(cv, splits.X_train),
        audit.audit_scaler_fit_on_train_only(),
        audit.audit_accuracy_sanity(test_summary["accuracy"], test_summary["majority_baseline_accuracy"]),
    ]
    audit_df = audit.format_audits(audits)

    # 9. persist machine-readable results
    perf.to_csv(out_dir / "backtest_performance.csv")
    sens.to_csv(out_dir / "fee_sensitivity.csv", index=False)
    perm.to_csv(out_dir / "permutation_importance_validation.csv")
    audit_df.to_csv(out_dir / "audit.csv", index=False)
    summary = {"config": cfg.to_dict(), "selected_model": best_name, "cv": cv_scores,
               "validation": val_rows, "test": test_summary, "backtest_engine": bh.engine,
               "backtest": perf.reset_index().to_dict("records"), "split_rows": splits.summary().astype(str).to_dict("records")}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    return dict(ds=ds, splits=splits, fitted=fitted, val_rows=val_rows, cv_scores=cv_scores, best_name=best_name,
                train_only=train_only, final=final, perm=perm, test_summary=test_summary, predictions=predictions,
                results=results, buy_hold=bh, perf=perf, sens=sens, audit=audit_df, X_live=X_live, summary=summary)
