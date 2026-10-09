"""Feature engineering.

LEAK-PROOF RULE: the feature row stamped with date *t* is computed ONLY from
OHLCV rows dated <= t (i.e. information available at the close of day t).
It is used to predict what happens AFTER t. Every operation below is causal:
pct_change(n), rolling(n), ewm(adjust=False) and shift(+k) never look forward.
`audit.audit_no_lookahead` proves this empirically with a truncation test.

Raw prices are non-stationary, so every feature is a return, a ratio, or a
bounded oscillator - never a raw price level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "returns": ["ret_1d", "ret_5d", "ret_10d", "ret_21d"],
    "trend": ["close_vs_sma10", "close_vs_sma50", "sma50_vs_sma200", "drawdown_252d"],
    "momentum": ["rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct"],
    "bollinger": ["bb_pctb_20", "bb_width_20"],
    "volatility": ["vol_10d", "vol_21d", "vol_63d", "vol_ratio_10_63", "atr_14_pct", "hl_range_pct"],
    "microstructure": ["close_in_range", "overnight_gap", "volume_ratio_20"],
}
FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return out


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (ohlcv[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    f = pd.DataFrame(index=ohlcv.index)
    r1 = c.pct_change()

    # --- returns (stationary version of price) -------------------------------
    f["ret_1d"] = r1
    for n in (5, 10, 21):
        f[f"ret_{n}d"] = c.pct_change(n)

    # --- trend / moving averages (as ratios, so they are scale-free) -----------
    sma10, sma50, sma200 = (c.rolling(n).mean() for n in (10, 50, 200))
    f["close_vs_sma10"] = c / sma10 - 1
    f["close_vs_sma50"] = c / sma50 - 1
    f["sma50_vs_sma200"] = sma50 / sma200 - 1
    f["drawdown_252d"] = c / c.rolling(252).max() - 1

    # --- momentum: RSI and MACD (normalised by price) --------------------------
    f["rsi_14"] = rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c
    f["macd_signal_pct"] = macd_sig / c
    f["macd_hist_pct"] = (macd - macd_sig) / c

    # --- Bollinger Bands (20, 2 sigma) ----------------------------------------
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    f["bb_pctb_20"] = (c - lower) / (upper - lower)
    f["bb_width_20"] = (upper - lower) / mid

    # --- historical volatility ---------------------------------------------------
    vol10, vol21, vol63 = (r1.rolling(n).std() for n in (10, 21, 63))
    f["vol_10d"], f["vol_21d"], f["vol_63d"] = vol10, vol21, vol63
    f["vol_ratio_10_63"] = vol10 / vol63
    prev_c = c.shift(1)
    true_range = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr_14_pct"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    f["hl_range_pct"] = (h - l) / c

    # --- intraday / volume context (all known at today's close) -------------------
    rng = (h - l).replace(0.0, np.nan)
    f["close_in_range"] = ((c - l) / rng).fillna(0.5)
    f["overnight_gap"] = o / prev_c - 1
    f["volume_ratio_20"] = v / v.rolling(20).mean() - 1

    f = f.replace([np.inf, -np.inf], np.nan)
    return f[FEATURE_NAMES]
