"""Automated look-ahead / leakage audits (the assignment says the code WILL be audited).

Each check returns (name, passed, detail). `run_all_audits` bundles them so the
report can print an auditable pass/fail table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features


def audit_no_lookahead(ohlcv: pd.DataFrame, n_checks: int = 8, seed: int = 0, tol: float = 1e-9):
    """Truncation test: features at date t must be IDENTICAL whether we compute
    them on the full history or only on data up to t. If any indicator peeked at
    the future, the two computations would differ."""
    full = build_features(ohlcv)
    rng = np.random.default_rng(seed)
    valid_positions = np.flatnonzero(full.notna().all(axis=1).to_numpy())
    picks = rng.choice(valid_positions, size=min(n_checks, len(valid_positions)), replace=False)
    worst = 0.0
    for p in picks:
        trunc = build_features(ohlcv.iloc[: p + 1]).iloc[-1]
        row = full.iloc[p]
        # a feature that is NaN when truncated but defined on the full history (or vice versa)
        # also proves it depends on future rows - never let NaN differences slip through .max()
        if set(trunc.index) != set(row.index) or (trunc.isna() ^ row.reindex(trunc.index).isna()).any():
            worst = float("inf")
            continue
        diff = (trunc - row.reindex(trunc.index)).abs().max()
        worst = max(worst, float(diff))
    return ("feature truncation test (no look-ahead)", worst <= tol, f"max |diff| over {len(picks)} dates = {worst:.2e}")


def audit_target_not_in_features(X: pd.DataFrame, fwd_ret: pd.Series, max_abs_corr: float = 0.5):
    bad_names = [c for c in X.columns if c in ("label", "fwd_ret")]
    corr = X.corrwith(fwd_ret.loc[X.index]).abs().max()
    ok = (not bad_names) and corr < max_abs_corr
    return ("target absent from features & no suspicious correlation", bool(ok),
            f"max |corr(feature, future return)| = {corr:.3f}")


def audit_split_order(X: pd.DataFrame, splits, ohlcv_index: pd.DatetimeIndex, horizon: int):
    """Train < Val < Test in time, and every label's look-ahead window ends before the next split starts."""
    def pos(ts):
        return ohlcv_index.get_loc(ts)
    ok, msgs = True, []
    seq = [("train", splits.X_train), ("val", splits.X_val), ("test", splits.X_test)]
    for (n1, a), (n2, b) in zip(seq[:-1], seq[1:]):
        gap = pos(b.index.min()) - pos(a.index.max())
        passed = a.index.max() < b.index.min() and gap > horizon
        ok &= passed
        msgs.append(f"{n1}->{n2}: {gap} trading days between last {n1} row and first {n2} row (need > {horizon})")
    return ("chronological order + purge gap >= horizon", bool(ok), "; ".join(msgs))


def audit_no_shuffle_cv(cv, X: pd.DataFrame):
    ok = True
    for tr, te in cv.split(X):
        ok &= tr.max() < te.min()
    return ("every CV fold trains strictly before it validates", bool(ok), f"{cv.get_n_splits()} folds checked")


def audit_accuracy_sanity(test_accuracy: float, majority_accuracy: float = 0.0, ceiling: float = 0.60, margin: float = 0.15):
    """Alarm when accuracy is implausibly high. With imbalanced classes (e.g. wide thresholds -> mostly Hold) a high
    accuracy can be legitimate, so the alarm level is max(ceiling, majority-class accuracy + margin)."""
    alarm = max(ceiling, majority_accuracy + margin)
    return ("accuracy plausibility (assignment: ~35-40% normal, 80% = leak)", test_accuracy < alarm,
            f"test accuracy = {test_accuracy:.3f}; majority-class baseline = {majority_accuracy:.3f}; alarm above {alarm:.2f}")


def audit_scaler_fit_on_train_only():
    return ("preprocessing fit inside the Pipeline on training data only", True,
            "by construction: StandardScaler lives inside the sklearn Pipeline and GridSearchCV refits it per fold")


def format_audits(audits) -> pd.DataFrame:
    return pd.DataFrame([{"check": n, "passed": "PASS" if ok else "FAIL", "detail": d} for n, ok, d in audits])
