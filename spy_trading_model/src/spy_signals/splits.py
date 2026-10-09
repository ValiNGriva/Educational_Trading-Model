"""Strictly chronological splits with PURGING.

Labels look `horizon` days into the future, so a training row dated within the
last `horizon` days of the training window has a label computed from prices
that belong to the validation window. Dropping those rows (purging) removes
this subtle leak. No random shuffling is ever used.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config


@dataclass
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    purged_train: int
    purged_val: int

    @property
    def X_trainval(self) -> pd.DataFrame:
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_trainval(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, X, y in (("train", self.X_train, self.y_train), ("validation", self.X_val, self.y_val),
                           ("test", self.X_test, self.y_test)):
            rows.append({"split": name, "start": X.index.min().date(), "end": X.index.max().date(),
                         "rows": len(X), **{f"share_{k}": round((y == k).mean(), 3) for k in (-1, 0, 1)}})
        return pd.DataFrame(rows)


def chronological_split(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Splits:
    h = cfg.horizon
    idx = X.index
    tr = (idx <= pd.Timestamp(cfg.train_end))
    va = (idx > pd.Timestamp(cfg.train_end)) & (idx <= pd.Timestamp(cfg.val_end))
    te = idx > pd.Timestamp(cfg.val_end)
    if min(tr.sum(), va.sum(), te.sum()) < 50:
        raise ValueError("A split has <50 rows - check start/end/train_end/val_end in Config.")

    def purge(mask):  # drop the last h rows of a window (labels leak into the next window)
        pos = mask.nonzero()[0]
        return pos[:-h] if h > 0 else pos

    tr_pos, va_pos, te_pos = purge(tr), purge(va), te.nonzero()[0]
    return Splits(
        X.iloc[tr_pos], y.iloc[tr_pos], X.iloc[va_pos], y.iloc[va_pos], X.iloc[te_pos], y.iloc[te_pos],
        purged_train=int(tr.sum() - len(tr_pos)), purged_val=int(va.sum() - len(va_pos)),
    )
