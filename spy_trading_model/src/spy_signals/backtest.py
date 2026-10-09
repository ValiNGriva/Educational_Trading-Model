"""Realistic backtest: model signals -> positions -> simulated orders with costs.

Timing (no look-ahead in execution either):
    * the model sees data through the CLOSE of day t and issues a signal,
    * the order is filled at the OPEN of day t+1,
    * every fill pays commission (`fee`) and adverse `slippage`,
    * equity is marked to market at each close.

Strategy is long-only (SPY ETF, no leverage, no shorting):
    Buy  -> go/stay long    Sell -> go to cash    Hold -> keep the current position

Engines:
    "vectorbt" - the dedicated backtesting library required by the assignment (default when installed)
    "builtin"  - a transparent ~40-line reference engine used for unit tests and as a
                 cross-check. Same timing/cost conventions as the vectorbt call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- signals
def signals_to_positions(pred: pd.Series, confidence: pd.Series | None = None, min_confidence: float = 0.0,
                         mode: str = "stateful") -> pd.Series:
    """Map class predictions (1/0/-1) to the DESIRED position (1=long, 0=cash) decided at each close.

    mode="stateful": Buy->long, Sell->cash, Hold->keep previous position (few trades).
    mode="daily"   : long only on days the model says Buy, else cash (trades on every flicker -
                     included to demonstrate the transaction-cost lesson).
    Signals with confidence < min_confidence are treated as Hold.
    """
    pred = pred.copy()
    if confidence is not None and min_confidence > 0:
        pred[confidence < min_confidence] = 0
    if mode == "daily":
        return (pred == 1).astype(int)
    pos, cur = [], 0
    for p in pred.to_numpy():
        if p == 1:
            cur = 1
        elif p == -1:
            cur = 0
        pos.append(cur)
    return pd.Series(pos, index=pred.index, name="position")


def to_orders(desired: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Desired position decided at close t becomes the held target on t+1 (filled at that open)."""
    target = desired.shift(1).fillna(0).astype(int)
    prev = target.shift(1).fillna(0).astype(int)
    entries = (target == 1) & (prev == 0)
    exits = (target == 0) & (prev == 1)
    return target, entries, exits


# --------------------------------------------------------------------------- results
@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    n_orders: int
    exposure: float
    engine: str
    init_cash: float
    raw: object = field(default=None, repr=False)   # vectorbt Portfolio when available

    def metrics(self) -> dict:
        eq = self.equity.to_numpy(dtype=float)
        path = np.r_[self.init_cash, eq]
        rets = path[1:] / path[:-1] - 1
        n = len(rets)
        total = path[-1] / path[0] - 1
        cagr = (path[-1] / path[0]) ** (TRADING_DAYS / max(n, 1)) - 1
        vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if n > 1 else np.nan
        sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if rets.std(ddof=1) > 0 else np.nan
        dd = (path / np.maximum.accumulate(path) - 1).min()
        return {"strategy": self.name, "final_value": float(path[-1]), "total_return": float(total),
                "cagr": float(cagr), "ann_vol": float(vol), "sharpe": float(sharpe),
                "max_drawdown": float(dd), "n_orders": int(self.n_orders), "exposure": float(self.exposure)}


# --------------------------------------------------------------------------- engines
def _builtin(open_, close, entries, exits, fee, slip, init_cash):
    cash, shares, n_orders = float(init_cash), 0.0, 0
    equity = np.empty(len(close))
    for i in range(len(close)):
        if entries.iloc[i] and shares == 0:
            px = open_.iloc[i] * (1 + slip)
            shares = cash / (px * (1 + fee))
            cash, n_orders = 0.0, n_orders + 1
        elif exits.iloc[i] and shares > 0:
            px = open_.iloc[i] * (1 - slip)
            cash = shares * px * (1 - fee)
            shares, n_orders = 0.0, n_orders + 1
        equity[i] = cash + shares * close.iloc[i]
    return pd.Series(equity, index=close.index), n_orders, None


def _vectorbt(open_, close, entries, exits, fee, slip, init_cash):
    import vectorbt as vbt

    pf = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, price=open_,
        fees=fee, slippage=slip, init_cash=init_cash, freq="1D",
    )
    equity = pf.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    return equity, int(pf.orders.count()), pf


def vectorbt_available() -> bool:
    try:
        import vectorbt  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_WARNED = False


def resolve_engine(engine: str) -> str:
    global _WARNED
    if engine == "auto":
        if vectorbt_available():
            return "vectorbt"
        if not _WARNED:
            _WARNED = True
            print("WARNING: vectorbt not installed - using the built-in reference engine. "
              "Install vectorbt (pip install -r requirements.txt) to satisfy the assignment's library requirement.")
        return "builtin"
    if engine == "vectorbt" and not vectorbt_available():
        raise ImportError("engine='vectorbt' requested but vectorbt is not installed.")
    return engine


def run_backtest(ohlcv: pd.DataFrame, desired_position: pd.Series, name: str, fee: float, slippage: float,
                 init_cash: float = 100_000.0, engine: str = "auto") -> BacktestResult:
    """`desired_position` is indexed by decision date (close). Prices are sliced to the same dates."""
    engine = resolve_engine(engine)
    px = ohlcv.loc[desired_position.index]
    target, entries, exits = to_orders(desired_position)
    fn = _vectorbt if engine == "vectorbt" else _builtin
    equity, n_orders, raw = fn(px["Open"], px["Close"], entries, exits, fee, slippage, init_cash)
    return BacktestResult(name, equity, n_orders, float(target.mean()), engine, init_cash, raw)


def buy_and_hold(ohlcv: pd.DataFrame, index: pd.DatetimeIndex, fee: float, slippage: float,
                 init_cash: float = 100_000.0, engine: str = "auto") -> BacktestResult:
    """Benchmark: buy SPY at the first open of the same window and hold (pays costs once)."""
    px = ohlcv.loc[index]
    engine = resolve_engine(engine)
    entries = pd.Series(False, index=index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=index)
    fn = _vectorbt if engine == "vectorbt" else _builtin
    equity, n_orders, raw = fn(px["Open"], px["Close"], entries, exits, fee, slippage, init_cash)
    return BacktestResult("Buy & Hold SPY", equity, n_orders, 1.0, engine, init_cash, raw)


def fee_sensitivity(ohlcv, desired: dict, fees, slippage, init_cash, engine="auto") -> pd.DataFrame:
    """Re-run every strategy under several commission levels - the critical fee lesson."""
    rows = []
    for fee in fees:
        bh = buy_and_hold(ohlcv, next(iter(desired.values())).index, fee, slippage, init_cash, engine)
        rows.append({**bh.metrics(), "fee": fee})
        for name, pos in desired.items():
            rows.append({**run_backtest(ohlcv, pos, name, fee, slippage, init_cash, engine).metrics(), "fee": fee})
    return pd.DataFrame(rows)[["fee", "strategy", "final_value", "total_return", "sharpe", "max_drawdown", "n_orders"]]
