"""Target definition: Buy (1) / Hold (0) / Sell (-1) from the FUTURE h-day return.

fwd_ret[t] = Close[t+h] / Close[t] - 1.  This deliberately looks forward - it is
the thing we predict - so it must never be used as a feature. The last h rows
have no future yet and get NaN labels (they are dropped from training/scoring,
but we still generate live signals for them in the backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_target(close: pd.Series, horizon: int = 5, up: float = 0.01, down: float = 0.01) -> pd.DataFrame:
    fwd = close.shift(-horizon) / close - 1
    label = pd.Series(0.0, index=close.index)
    label[fwd > up] = 1.0
    label[fwd < -down] = -1.0
    label[fwd.isna()] = np.nan
    return pd.DataFrame({"fwd_ret": fwd, "label": label})


def class_distribution(y: pd.Series, names: dict | None = None) -> pd.DataFrame:
    names = names or {-1: "Sell", 0: "Hold", 1: "Buy"}
    counts = y.value_counts().reindex([-1, 0, 1]).fillna(0).astype(int)
    out = pd.DataFrame({"class": [names[i] for i in counts.index], "count": counts.values})
    out["share"] = out["count"] / out["count"].sum()
    return out
