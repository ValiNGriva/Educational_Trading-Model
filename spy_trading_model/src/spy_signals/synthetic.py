"""Synthetic SPY-like data. ONLY for smoke-testing the code path offline.

Nothing learned from this data says anything about the real market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(seed)
    p_stay = (0.995, 0.97)          # calm regime is sticky, crisis regime is short
    mu = (0.0006, -0.0005)
    sig = (0.0065, 0.019)
    state, r = 0, np.empty(n)
    for i in range(n):
        if rng.random() > p_stay[state]:
            state = 1 - state
        shock = rng.standard_t(5) / np.sqrt(5 / 3)
        r[i] = mu[state] + sig[state] * shock
    close = 100 * np.cumprod(1 + r)
    prev = np.r_[close[0], close[:-1]]
    open_ = prev * (1 + rng.normal(0, 0.0025, n))
    spread = np.abs(rng.normal(0, 0.004, n)) + np.abs(r) * 0.5
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = (rng.lognormal(17, 0.25, n) * (1 + 15 * np.abs(r))).astype("int64")
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df
