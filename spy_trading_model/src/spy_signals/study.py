"""Multi-model study used by the Streamlit dashboard (no Streamlit imports here, so it is unit-tested).

Two-speed design:
  * `train_all`      - expensive; depends on data + thresholds + split dates. Run once, cache.
  * `backtest_stage` - cheap; depends on fee / slippage / trading rule. Re-run on every slider move.

Stages
  * validation: every model tuned on TRAIN only is scored on the validation window
                (classification metrics + a validation-window backtest, in-sample for model selection).
  * test      : every model re-fit on TRAIN+VALIDATION, scored once on the untouched test window
                (classification metrics + the realistic backtest).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import audit, backtest, evaluate, explain, models
from .config import Config
from .pipeline import Dataset, prepare_data
from .splits import Splits, chronological_split

MIN_ROWS_PER_CLASS = 10   # a class needs at least this many TRAIN rows to count as "present"
CLASS_COLS = ["Sell", "Hold", "Buy"]


class StudyError(RuntimeError):
    """Raised with a user-readable message when the chosen settings cannot be modelled."""


@dataclass
class ModelRun:
    name: str
    kind: str                       # "baseline" | "model"
    cv_score: float | None
    best_params: dict | None
    val_summary: dict
    test_summary: dict
    val_pred: pd.DataFrame          # Sell/Hold/Buy probs + pred + confidence, every date in the validation window
    test_pred: pd.DataFrame         # same for the test window
    train_only_model: object
    final_model: object
    note: str = ""


@dataclass
class Study:
    cfg: Config
    ohlcv: pd.DataFrame
    ds: Dataset
    splits: Splits
    runs: dict
    audits: pd.DataFrame
    best_name: str
    _perm: dict = field(default_factory=dict)

    @property
    def model_names(self):
        return [n for n, r in self.runs.items() if r.kind == "model"]


# ----------------------------------------------------------------------------- feasibility
def class_feasibility(y_train: pd.Series) -> dict:
    counts = {n: int((y_train == c).sum()) for n, c in (("Sell", -1), ("Hold", 0), ("Buy", 1))}
    present = [n for n, k in counts.items() if k >= MIN_ROWS_PER_CLASS]
    return {"counts": counts, "present": present, "ok": len(present) >= 2}


# ----------------------------------------------------------------------------- helpers
def _predict_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilities for ALL three classes (absent classes get 0), plus pred + confidence."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    proba = pd.DataFrame(model.predict_proba(X), index=X.index,
                         columns=[evaluate.NAMES[int(c)] for c in clf.classes_])
    proba = proba.reindex(columns=CLASS_COLS, fill_value=0.0)
    out = proba.copy()
    out["pred"] = model.predict(X).astype(int)
    out["confidence"] = proba.max(axis=1)
    return out


def _fit_tuned(name, est, grid, splits, cfg):
    """Tune with purged TimeSeriesSplit; if CV is impossible (e.g. a class missing in every fold) fall back
    to default parameters and say so, rather than crashing the dashboard."""
    try:
        s = models.tune(name, est, grid, splits.X_train, splits.y_train, cfg)
        if not np.isfinite(s.best_score_):
            raise ValueError("non-finite CV score")
        return s.best_estimator_, float(s.best_score_), dict(s.best_params_), ""
    except Exception as exc:  # noqa: BLE001
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = clone(est).fit(splits.X_train, splits.y_train)
        return fit, None, None, f"CV unavailable ({type(exc).__name__}); default parameters used."


# ----------------------------------------------------------------------------- training
def train_all(cfg: Config, ohlcv: pd.DataFrame, progress=None) -> Study:
    """Fit baselines + all models for the validation stage and the test stage."""
    ds = prepare_data(cfg, ohlcv)
    splits = chronological_split(ds.X, ds.y, cfg)
    feas = class_feasibility(splits.y_train)
    if not feas["ok"]:
        raise StudyError(
            f"With +{cfg.up_threshold:.1%} / -{cfg.down_threshold:.1%} thresholds over {cfg.horizon} days the training "
            f"period has only these labelled days: {feas['counts']}. Fewer than two classes have >= {MIN_ROWS_PER_CLASS} "
            "examples, so there is nothing to classify. Lower the thresholds.")

    X_live_val = ds.X_all.loc[(ds.X_all.index > pd.Timestamp(cfg.train_end)) & (ds.X_all.index <= pd.Timestamp(cfg.val_end))]
    X_live_test = ds.X_all.loc[ds.X_all.index > pd.Timestamp(cfg.val_end)]

    jobs = [(n, "baseline", e, None) for n, e in models.baseline_models().items()]
    jobs += [(n, "model", e, g) for n, (e, g) in models.model_zoo(cfg).items()]
    runs = {}
    for i, (name, kind, est, grid) in enumerate(jobs):
        if progress:
            progress(i / len(jobs), f"Training {name} ...")
        if kind == "baseline":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_only = clone(est).fit(splits.X_train, splits.y_train)
            cv_score, params, note = None, None, ""
        else:
            train_only, cv_score, params, note = _fit_tuned(name, est, grid, splits, cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = clone(train_only).fit(splits.X_trainval, splits.y_trainval)   # refit on train+val (purged)
            val_summary = evaluate.classification_summary(splits.y_val, train_only.predict(splits.X_val))
            test_summary = evaluate.classification_summary(splits.y_test, final.predict(splits.X_test))
        val_summary["majority_baseline_accuracy"] = evaluate.majority_accuracy(splits.y_train, splits.y_val)
        test_summary["majority_baseline_accuracy"] = evaluate.majority_accuracy(splits.y_trainval, splits.y_test)
        runs[name] = ModelRun(name, kind, cv_score, params, val_summary, test_summary,
                              _predict_frame(train_only, X_live_val), _predict_frame(final, X_live_test),
                              train_only, final, note)
    if progress:
        progress(1.0, "Done")

    real = [n for n, r in runs.items() if r.kind == "model"]
    best = max(real, key=lambda n: runs[n].val_summary["macro_f1"])
    audits = audit.format_audits([
        audit.audit_no_lookahead(ohlcv),
        audit.audit_target_not_in_features(ds.X, ds.fwd_ret),
        audit.audit_split_order(ds.X, splits, ohlcv.index, cfg.horizon),
        audit.audit_no_shuffle_cv(models.time_series_cv(cfg), splits.X_train),
        audit.audit_scaler_fit_on_train_only(),
        audit.audit_accuracy_sanity(runs[best].test_summary["accuracy"], runs[best].test_summary["majority_baseline_accuracy"]),
    ])
    return Study(cfg, ohlcv, ds, splits, runs, audits, best)


# ----------------------------------------------------------------------------- tables
def classification_table(study: Study, stage: str) -> pd.DataFrame:
    """stage: 'validation' | 'test'. One row per model with every classification metric."""
    rows = {}
    for n, r in study.runs.items():
        s = r.val_summary if stage == "validation" else r.test_summary
        row = {"kind": r.kind, "accuracy": s["accuracy"], "balanced_accuracy": s["balanced_accuracy"],
               "macro_f1": s["macro_f1"], "majority_baseline_acc": s["majority_baseline_accuracy"]}
        for c in CLASS_COLS:
            row[f"recall_{c}"] = s["per_class"][c]["recall"]
        if stage == "validation":
            row["cv_macro_f1(train)"] = r.cv_score
        rows[n] = row
    return pd.DataFrame(rows).T.infer_objects()


def permutation_for(study: Study, name: str, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance of the TRAIN-only model on the validation window (memoised)."""
    if name not in study._perm:
        r = study.runs[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study._perm[name] = explain.permutation_table(r.train_only_model, study.splits.X_val, study.splits.y_val,
                                                          study.cfg.scoring, n_repeats, study.cfg.random_state)
    return study._perm[name]


# ----------------------------------------------------------------------------- backtests
def _window(study: Study, stage: str) -> pd.DatetimeIndex:
    return (study.runs[next(iter(study.runs))].val_pred if stage == "validation"
            else study.runs[next(iter(study.runs))].test_pred).index


def desired_positions(study: Study, stage: str, rule: str, min_confidence: float) -> dict:
    out = {}
    for n, r in study.runs.items():
        p = r.val_pred if stage == "validation" else r.test_pred
        out[n] = backtest.signals_to_positions(p["pred"], p["confidence"], min_confidence, rule)
    return out


def backtest_stage(study: Study, stage: str, fee: float, slippage: float, rule: str = "stateful",
                   min_confidence: float = 0.0, engine: str = "auto") -> dict:
    """Backtest EVERY model on the validation or test window under the given costs."""
    cfg = study.cfg
    idx = _window(study, stage)
    bh = backtest.buy_and_hold(study.ohlcv, idx, fee, slippage, cfg.init_cash, engine)
    results = {n: backtest.run_backtest(study.ohlcv, pos, n, fee, slippage, cfg.init_cash, engine)
               for n, pos in desired_positions(study, stage, rule, min_confidence).items()}
    perf = pd.DataFrame([bh.metrics()] + [r.metrics() for r in results.values()]).set_index("strategy")
    return {"perf": perf, "results": results, "bh": bh, "index": idx}


def fee_sweep(study: Study, stage: str, names: list, fees: list, slippage: float, rule: str = "stateful",
              min_confidence: float = 0.0, engine: str = "auto") -> pd.DataFrame:
    """Re-run selected models across commission levels (fee column is a fraction, e.g. 0.02 = 2%)."""
    cfg = study.cfg
    pos = {n: p for n, p in desired_positions(study, stage, rule, min_confidence).items() if n in names}
    idx = _window(study, stage)
    rows = []
    for fee in fees:
        bh = backtest.buy_and_hold(study.ohlcv, idx, fee, slippage, cfg.init_cash, engine)
        rows.append({**bh.metrics(), "fee": fee})
        for n, p in pos.items():
            rows.append({**backtest.run_backtest(study.ohlcv, p, n, fee, slippage, cfg.init_cash, engine).metrics(), "fee": fee})
    return pd.DataFrame(rows)


def report_inputs(study: Study, fee: float, slippage: float, engine: str = "auto", min_confidence: float = 0.0) -> dict:
    """Assemble the dict `report.build_report` expects, for the validation-selected model."""
    cfg = Config(**{**study.cfg.to_dict(), "fee": fee, "slippage": slippage, "engine": engine,
                    "min_confidence": min_confidence})
    best = study.runs[study.best_name]
    fees = sorted({0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, fee})
    pos = {"Model (stateful)": backtest.signals_to_positions(best.test_pred["pred"], best.test_pred["confidence"], min_confidence, "stateful"),
           "Model (daily Buy-only)": backtest.signals_to_positions(best.test_pred["pred"], best.test_pred["confidence"], min_confidence, "daily")}
    results = {n: backtest.run_backtest(study.ohlcv, p, n, fee, slippage, cfg.init_cash, engine) for n, p in pos.items()}
    idx = best.test_pred.index
    bh = backtest.buy_and_hold(study.ohlcv, idx, fee, slippage, cfg.init_cash, engine)
    perf = pd.DataFrame([bh.metrics()] + [r.metrics() for r in results.values()]).set_index("strategy")
    sens = backtest.fee_sensitivity(study.ohlcv, pos, fees, slippage, cfg.init_cash, engine)
    return dict(cfg=cfg, ds=study.ds, splits=study.splits, best_name=study.best_name,
                val_rows={n: r.val_summary for n, r in study.runs.items()}, test_summary=best.test_summary,
                perm=permutation_for(study, study.best_name), perf=perf, sens=sens, audit=study.audits, buy_hold=bh)
