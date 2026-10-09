"""Auto-generates the 'Lessons Learned' report from the ACTUAL results.

Numbers are filled in from the run; the interpretive sentences are chosen by
rules on those numbers (e.g. did the model beat buy & hold? how much did fees
cost?). Always read it critically and add your own analysis in the final section.
"""
from __future__ import annotations

import pandas as pd


def _pct(x):
    return f"{x * 100:.1f}%"


def _pct2(x):
    return f"{x * 100:.2f}%"


def build_report(cfg, res: dict, data_source_note: str = "") -> str:
    t, v = res["test_summary"], res["val_rows"][res["best_name"]]
    perf, sens = res["perf"], res["sens"]
    m, d, bh = perf.loc["Model (stateful)"], perf.loc["Model (daily Buy-only)"], perf.loc["Buy & Hold SPY"]
    pc = t["per_class"]
    shares = {k: float((res["splits"].y_test == c).mean()) for k, c in (("Sell", -1), ("Hold", 0), ("Buy", 1))}
    maj = t["majority_baseline_accuracy"]
    free = sens[(sens["fee"] == 0.0) & (sens["strategy"] == "Model (stateful)")].iloc[0]
    paid = sens[(sens["fee"] == cfg.fee) & (sens["strategy"] == "Model (stateful)")]
    fee_drag = (free["final_value"] - m["final_value"]) / cfg.init_cash if len(paid) else float("nan")
    free_d = sens[(sens["fee"] == 0.0) & (sens["strategy"] == "Model (daily Buy-only)")].iloc[0]
    fee_drag_d = (free_d["final_value"] - d["final_value"]) / cfg.init_cash

    L = []
    a = L.append
    a(f"# Lessons Learned - {cfg.ticker} {cfg.horizon}-day Buy/Sell/Hold model\n")
    if data_source_note:
        a(f"> **{data_source_note}**\n")
    a("## 1. Setup in one paragraph\n")
    a(f"Daily {cfg.ticker} OHLCV from {cfg.start} to {cfg.end}. {len(res['ds'].X.columns)} stationary features "
      f"(returns, SMA ratios, RSI, MACD, Bollinger, rolling volatility, ATR, volume context), all computed from data up to "
      f"the close of day *t*. Target: Buy if the next {cfg.horizon}-day return > +{_pct(cfg.up_threshold)}, Sell if "
      f"< -{_pct(cfg.down_threshold)}, else Hold. Split: train through {cfg.train_end}, validation through {cfg.val_end}, "
      f"test afterwards, with a {cfg.horizon}-day purge at each boundary. Hyper-parameters were tuned with a purged "
      f"`TimeSeriesSplit`; imbalance handled with class weights. Selected model: **{res['best_name']}**. "
      f"Backtest engine: `{res['buy_hold'].engine}`, fee {_pct2(cfg.fee)} + slippage {_pct2(cfg.slippage)} per order, "
      f"signals decided at the close and filled at the next open.\n")

    a("## 2. Headline numbers (test period, touched once)\n")
    a(f"* Accuracy **{t['accuracy']:.3f}** vs. majority-class baseline **{maj:.3f}** and 3-class chance 0.333; "
      f"balanced accuracy {t['balanced_accuracy']:.3f}, macro-F1 {t['macro_f1']:.3f} "
      f"(validation macro-F1 was {v['macro_f1']:.3f}).")
    a(f"* Test class mix: Sell {_pct(shares['Sell'])}, Hold {_pct(shares['Hold'])}, Buy {_pct(shares['Buy'])}.")
    a(f"* Recall - Sell {pc['Sell']['recall']:.2f}, Hold {pc['Hold']['recall']:.2f}, Buy {pc['Buy']['recall']:.2f}.\n")
    tbl = perf[["final_value", "total_return", "cagr", "sharpe", "max_drawdown", "n_orders", "exposure"]].copy()
    for c in ("total_return", "cagr", "max_drawdown", "exposure"):
        tbl[c] = tbl[c].map(_pct)
    tbl["final_value"] = tbl["final_value"].map(lambda x: f"${x:,.0f}")
    tbl["sharpe"] = tbl["sharpe"].map(lambda x: f"{x:.2f}")
    a(tbl.to_markdown() if hasattr(tbl, "to_markdown") and _has_tabulate() else "```\n" + tbl.to_string() + "\n```")
    a("")

    a("## 3. Where did the model fail?\n")
    if t["accuracy"] <= maj + 0.01:
        a(f"* Accuracy ({t['accuracy']:.3f}) does **not** beat simply predicting the majority class ({maj:.3f}). "
          "Raw accuracy is therefore not evidence of skill here; judge it by balanced accuracy and per-class recall.")
    else:
        a(f"* Accuracy edges past the majority baseline ({t['accuracy']:.3f} vs {maj:.3f}), but the margin is small "
          "relative to the sampling noise of a few hundred overlapping-label days.")
    weakest = min(pc, key=lambda k: pc[k]["recall"])
    a(f"* Weakest class is **{weakest}** (recall {pc[weakest]['recall']:.2f}, {pc[weakest]['support']} test days).")
    if weakest == "Sell":
        a("  Hypothesis (not tested here): Sell days cluster in sudden sell-offs driven by news/liquidity that "
          "price-and-volume indicators only reflect after the fact.")
    top = res["perm"].head(3).index.tolist()
    vol_like = ("vol_", "atr_", "hl_range", "bb_width")
    if all(f.startswith(vol_like) for f in top):
        a(f"* The most-used features on validation were {', '.join(top)} - all volatility measures. Volatility is "
          "persistent and tells you how *large* a move is likely to be (Hold vs. Buy/Sell) much more reliably than its "
          "direction, which is consistent with weak directional skill.")
    else:
        a(f"* The most-used features on validation were {', '.join(top)} (permutation importance, drop in macro-F1). "
          "Check the importance chart: importances close to zero mean the model barely relies on any single feature.")
    gap = v["macro_f1"] - t["macro_f1"]
    if abs(gap) >= 0.05:
        a(f"* Validation vs. test macro-F1 ({v['macro_f1']:.3f} -> {t['macro_f1']:.3f}) differ by {abs(gap):.3f}. "
          "A gap this size means performance does not transfer cleanly across periods (regime change and/or "
          "model selection on a small validation window).\n")
    else:
        a(f"* Validation and test macro-F1 are close ({v['macro_f1']:.3f} vs {t['macro_f1']:.3f}), so there is no "
          "sign of heavy selection overfitting - but also remember both may simply be near chance.\n")

    a("## 4. Why it under/over-performed Buy & Hold\n")
    if m["final_value"] >= bh["final_value"]:
        a(f"* The stateful model ended at ${m['final_value']:,.0f} vs ${bh['final_value']:,.0f} for Buy & Hold "
          f"({_pct(m['total_return'])} vs {_pct(bh['total_return'])}). Before crediting skill, check risk: "
          f"max drawdown {_pct(m['max_drawdown'])} vs {_pct(bh['max_drawdown'])}, Sharpe {m['sharpe']:.2f} vs "
          f"{bh['sharpe']:.2f}, exposure {_pct(m['exposure'])}. One test window is one draw from history - treat "
          "outperformance as a hypothesis to re-test, not a result.")
    else:
        a(f"* The stateful model ended at ${m['final_value']:,.0f} vs ${bh['final_value']:,.0f} for Buy & Hold "
          f"({_pct(m['total_return'])} vs {_pct(bh['total_return'])}).")
        if m["exposure"] < 0.9:
            a(f"* **Time out of the market.** The model was invested only {_pct(m['exposure'])} of days. SPY drifts up, so "
              "every day in cash forgoes that drift; the model must be right about *when* to be out just to break even.")
    a(f"* **Costs matter.** Commission alone ({_pct2(cfg.fee)} per order, slippage held constant) cost the stateful "
      f"model about {_pct(fee_drag)} of starting capital over {int(m['n_orders'])} orders. The daily Buy-only variant "
      f"placed {int(d['n_orders'])} orders and lost about {_pct(fee_drag_d)} of capital to commission "
      f"(final ${d['final_value']:,.0f}). See the fee-sensitivity table: the more a strategy trades, the faster its "
      "edge (if any) is consumed.")
    a(f"* **Label overlap.** {cfg.horizon}-day labels on consecutive days share {cfg.horizon - 1} of {cfg.horizon} days of returns, so the {len(res['splits'].y_test)} "
      "test rows carry far fewer independent observations than that; metrics have wide error bars.\n")

    a("## 5. Why predicting markets is hard\n")
    a("* **Low signal-to-noise.** Daily index moves are dominated by news that is unpredictable from past prices; "
      "any exploitable pattern in public indicators is competed away.")
    a(f"* **Non-stationarity.** Relationships fitted on data through {cfg.train_end} need not hold afterwards - "
      "market regimes (rates, volatility, liquidity) change, so out-of-sample scores can drift away from in-sample ones.")
    a("* **Class imbalance reflects the market,** not just the data: upward drift tends to make Buy and Hold more common "
      "than Sell (see the class-distribution chart), so naive accuracy can reward doing nothing.")
    a("* **Theory vs. execution.** A model can classify decently and still lose money after costs, delayed fills and "
      "the cost of sitting out - the central lesson of this project.\n")

    a("## 6. Integrity checks\n")
    a("```\n" + res["audit"].to_string(index=False) + "\n```\n")

    a("## 7. Limitations and next steps (edit with your own analysis)\n")
    a("* Single test window; extend with walk-forward evaluation (repeated expanding-window refits).")
    a("* Try volatility-scaled thresholds, a minimum-confidence filter (`Config.min_confidence`), or a minimum holding period "
      "to cut turnover.")
    a("* Add exogenous features (VIX, yields, breadth) - the lesson-06 point that exogenous signals are where ML "
      "forecasting can add value over pure price history.")
    a("* Replace the single backtest with a distribution: bootstrap or re-start the strategy on many windows.\n")
    return "\n".join(L)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
