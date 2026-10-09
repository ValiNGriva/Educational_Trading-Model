"""Data loading with validation. Synthetic data is never used silently."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import Config
from .synthetic import make_synthetic_ohlcv

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")
    df = df[REQUIRED].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna()
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive prices found.")
    if len(df) < 1000:
        raise ValueError(f"Only {len(df)} rows - need several years of daily data.")
    return df


def _download_yfinance(cfg: Config) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so offline use still works

    end = (pd.Timestamp(cfg.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive
    df = yf.download(cfg.ticker, start=cfg.start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data (offline or ticker problem).")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (Price, Ticker)
        df.columns = df.columns.get_level_values(0)
    return df


def load_ohlcv(cfg: Config, source: str | None = None) -> pd.DataFrame:
    """Load daily OHLCV.

    source: "auto" (cache -> yfinance), "yfinance" (force refresh), "csv", or "synthetic".
    The env var SPY_DATA_SOURCE overrides the default so notebooks can be switched
    without editing code.
    """
    source = (source or os.environ.get("SPY_DATA_SOURCE") or "auto").lower()
    cache = Path(cfg.data_dir) / f"{cfg.ticker}_ohlcv.csv"
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        print("!" * 70)
        print("USING SYNTHETIC DATA - pipeline smoke test only, NOT real SPY results.")
        print("!" * 70)
        return _validate(make_synthetic_ohlcv(cfg.start, cfg.end, cfg.random_state))

    if source in ("auto", "csv") and cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"Loaded cached data: {cache}  ({len(df)} rows)")
        return _validate(df)
    if source == "csv":
        raise FileNotFoundError(f"{cache} not found. Put a CSV with Date,Open,High,Low,Close,Volume there.")

    try:
        df = _validate(_download_yfinance(cfg))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download {cfg.ticker} data ({exc}).\n"
            f"Options: (1) check your internet connection, (2) drop a CSV at {cache}, or "
            f"(3) run with --source synthetic for a code-path smoke test."
        ) from exc
    df.to_csv(cache)
    print(f"Downloaded {len(df)} rows from yfinance and cached to {cache}")
    return df
