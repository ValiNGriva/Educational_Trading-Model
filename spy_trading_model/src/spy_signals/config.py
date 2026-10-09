"""Single source of truth for every experiment parameter."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    ticker: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2025-12-31"
    data_dir: str = str(ROOT / "data")
    output_dir: str = str(ROOT / "outputs")

    # ---- target -----------------------------------------------------------
    horizon: int = 5            # trading days ahead
    up_threshold: float = 0.01  # Buy  if fwd return >  +1.0 %
    down_threshold: float = 0.01  # Sell if fwd return < -1.0 %

    # ---- chronological splits (inclusive end dates) -------------------------
    train_end: str = "2020-12-31"   # Train: start .. 2020
    val_end: str = "2022-12-31"     # Validation: 2021-2022; Test: 2023 onward

    # ---- tuning -----------------------------------------------------------
    n_cv_splits: int = 5
    scoring: str = "f1_macro"       # treats Buy/Sell/Hold equally
    random_state: int = 42
    fast: bool = False              # smaller grids (used by tests / quick demos)

    # ---- backtest -----------------------------------------------------------
    engine: str = "auto"            # "auto" | "vectorbt" | "builtin"
    init_cash: float = 100_000.0
    fee: float = 0.001              # 0.10 % commission per order
    slippage: float = 0.0005        # 0.05 % adverse fill per order
    min_confidence: float = 0.0     # ignore signals whose top-class prob is lower

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def labels(self) -> tuple:
        return (-1, 0, 1)

    @property
    def label_names(self) -> dict:
        return {-1: "Sell", 0: "Hold", 1: "Buy"}
