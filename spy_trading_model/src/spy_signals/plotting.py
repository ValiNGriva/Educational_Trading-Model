"""All figures (matplotlib only). Each function returns the Figure; pass `path` to save."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import NAMES

COL = {-1: "#c0392b", 0: "#7f8c8d", 1: "#27ae60"}


def _save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_price_with_splits(ohlcv, cfg, path=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ohlcv.index, ohlcv["Close"], lw=1, color="#2c3e50")
    ax.axvspan(ohlcv.index.min(), pd.Timestamp(cfg.train_end), color="#3498db", alpha=0.12, label="Train")
    ax.axvspan(pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.val_end), color="#f39c12", alpha=0.18, label="Validation")
    ax.axvspan(pd.Timestamp(cfg.val_end), ohlcv.index.max(), color="#27ae60", alpha=0.15, label="Test")
    ax.set(title=f"{cfg.ticker} close and chronological splits", ylabel="Adj. close ($)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_class_distribution(y_by_split: dict, path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    for i, (name, y) in enumerate(y_by_split.items()):
        shares = [(y == k).mean() for k in (-1, 0, 1)]
        ax.bar(np.arange(3) + i * w, shares, w, label=name)
    ax.set_xticks(np.arange(3) + w * (len(y_by_split) - 1) / 2, [NAMES[k] for k in (-1, 0, 1)])
    ax.set(ylabel="share of days", title="Target class distribution by split")
    ax.legend()
    return _save(fig, path)


def plot_threshold_sensitivity(table: pd.DataFrame, path=None):
    ax = table.plot(kind="bar", stacked=True, figsize=(8, 4), color=[COL[-1], COL[0], COL[1]], rot=0)
    ax.set(title="Class balance vs. threshold choice (+/- x%)", ylabel="share of days", xlabel="threshold")
    return _save(ax.figure, path)


def plot_feature_distributions(X: pd.DataFrame, path=None, ncols=5):
    n = X.shape[1]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.3 * nrows))
    for ax, c in zip(axes.ravel(), X.columns):
        s = X[c].clip(X[c].quantile(0.005), X[c].quantile(0.995))
        ax.hist(s, bins=40, color="#34495e", alpha=0.85)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions (winsorised at 0.5/99.5 pct for display)", y=1.0)
    return _save(fig, path)


def plot_correlation(X: pd.DataFrame, path=None):
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Feature correlation matrix")
    return _save(fig, path)


def plot_feature_vs_target(X: pd.DataFrame, y: pd.Series, path=None, top=12):
    """Mean of each (standardised) feature per class - shows how weak the signal is."""
    z = (X - X.mean()) / X.std()
    means = z.groupby(y).mean().T
    order = means.abs().max(axis=1).sort_values(ascending=False).index[:top]
    ax = means.loc[order].rename(columns=NAMES).plot(kind="barh", figsize=(8, 5.5), color=[COL[-1], COL[0], COL[1]])
    ax.set(title="Class-conditional feature means (z-scores): small gaps = weak signal", xlabel="mean z-score")
    ax.invert_yaxis()
    return _save(ax.figure, path)


def plot_forward_return_hist(fwd: pd.Series, cfg, path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fwd.clip(-0.12, 0.12), bins=80, color="#34495e")
    for t, c in ((-cfg.down_threshold, COL[-1]), (cfg.up_threshold, COL[1])):
        ax.axvline(t, color=c, ls="--")
    ax.set(title=f"{cfg.horizon}-day forward return and Buy/Sell thresholds", xlabel="forward return")
    return _save(fig, path)


def plot_confusion(cm, title, path=None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    names = [NAMES[k] for k in (-1, 0, 1)]
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set(xlabel="predicted", ylabel="actual", title=title + "\n(shade = row-normalised recall)")
    return _save(fig, path)


def plot_importance(perm: pd.DataFrame, path=None, top=15):
    p = perm.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(p.index, p["importance_mean"], xerr=p["importance_std"], color="#2980b9")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(title="Permutation importance (validation, drop in macro-F1)", xlabel="score drop when feature shuffled")
    return _save(fig, path)


def plot_logreg_coefs(coefs: pd.DataFrame, path=None, top=12):
    order = coefs.abs().max(axis=1).sort_values(ascending=False).index[:top]
    ax = coefs.loc[order].plot(kind="barh", figsize=(8, 5.5), color=[COL[-1], COL[0], COL[1]])
    ax.invert_yaxis()
    ax.set(title="Logistic-regression coefficients (standardised features)")
    return _save(ax.figure, path)


def plot_equity(results: list, path=None, title: str | None = None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        lw = 2.2 if r.name.startswith("Buy") else 1.5
        a1.plot(r.equity.index, r.equity, label=r.name, lw=lw)
        dd = r.equity / r.equity.cummax() - 1
        a2.plot(dd.index, dd, lw=1.2)
    a1.set(title=title or "Portfolio value: model signals (after fees + slippage) vs. Buy & Hold", ylabel="Portfolio value ($)")
    a1.legend()
    a2.set(ylabel="Drawdown")
    return _save(fig, path)


def plot_fee_sensitivity(sens: pd.DataFrame, path=None):
    piv = sens.pivot(index="fee", columns="strategy", values="final_value")
    ax = piv.plot(marker="o", figsize=(8, 4.5))
    ax.set(title="Final portfolio value vs. commission per order", ylabel="Final value ($)", xlabel="fee per order")
    return _save(ax.figure, path)


def plot_metric_bars(df: pd.DataFrame, title: str, ylabel: str, path=None, pct=False):
    """Grouped bars: rows = models, columns = stages (e.g. Validation vs Test)."""
    ax = df.plot(kind="bar", figsize=(9, 4.2), rot=20)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=title, ylabel=ylabel)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _save(ax.figure, path)


def plot_fee_sweep(sweep: pd.DataFrame, value_col: str, title: str, path=None, pct=True):
    """sweep columns: fee, strategy, <value_col>. One line per strategy across commission levels."""
    piv = sweep.pivot(index="fee", columns="strategy", values=value_col)
    ax = piv.plot(marker="o", figsize=(9, 4.5))
    ax.set(title=title, xlabel="commission per order", ylabel=value_col.replace("_", " "))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2%}")
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _save(ax.figure, path)
