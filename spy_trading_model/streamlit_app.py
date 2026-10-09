"""Interactive dashboard for the SPY Buy/Sell/Hold project.

Run:  streamlit run streamlit_app.py

Two-speed interaction (so the app stays responsive):
  * Sidebar form "Data & target"  -> thresholds (up to 20 %), data source, split dates. Needs "Apply & train"
                                     because changing the label definition re-tunes all models.
  * Sidebar "Trading costs"       -> fee (up to 2 % per trade), slippage, trading rule. Instant: only the cheap
                                     backtest is recomputed; models are NOT retrained.
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from spy_signals import Config, evaluate, explain, plotting as P, study as S  # noqa: E402
from spy_signals.data import load_ohlcv  # noqa: E402
from spy_signals.features import FEATURE_GROUPS  # noqa: E402
from spy_signals.pipeline import prepare_data  # noqa: E402
from spy_signals.report import build_report  # noqa: E402
from spy_signals.splits import chronological_split  # noqa: E402
from spy_signals.target import class_distribution  # noqa: E402

st.set_page_config(page_title="SPY Trading Signal Lab", layout="wide")

DEFAULTS = dict(source="auto", ticker="SPY", start=date(2010, 1, 1), end=date(2025, 12, 31), up=1.0, down=1.0,
                linked=True, train_end=date(2020, 12, 31), val_end=date(2022, 12, 31), fast=True)
HORIZON = 5


# =============================================================================== helpers
def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show(fig):
    st.image(fig_png(fig))


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def perf_display(perf: pd.DataFrame) -> pd.DataFrame:
    d = perf.copy()
    d["final_value"] = d["final_value"].map(lambda v: f"${v:,.0f}")
    for c in ("total_return", "cagr", "ann_vol", "max_drawdown", "exposure"):
        d[c] = d[c].map(pct)
    d["sharpe"] = d["sharpe"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    d["n_orders"] = d["n_orders"].astype(int)
    return d


# =============================================================================== cached computation
@st.cache_resource(show_spinner=False)
def get_ohlcv(source, ticker, start, end):
    cfg = Config(ticker=ticker, start=start, end=end)
    df = load_ohlcv(cfg, source)
    return df.loc[start:end]            # a cached CSV may cover a wider range than requested


@st.cache_resource(show_spinner=False)
def get_prepared(source, ticker, start, end, up, down, train_end, val_end):
    ohlcv = get_ohlcv(source, ticker, start, end)
    cfg = Config(ticker=ticker, start=start, end=end, up_threshold=up, down_threshold=down, horizon=HORIZON,
                 train_end=train_end, val_end=val_end)
    ds = prepare_data(cfg, ohlcv)
    return cfg, ohlcv, ds, chronological_split(ds.X, ds.y, cfg)


@st.cache_resource(show_spinner=False)
def get_study(source, ticker, start, end, up, down, train_end, val_end, fast, seed):
    ohlcv = get_ohlcv(source, ticker, start, end)
    cfg = Config(ticker=ticker, start=start, end=end, up_threshold=up, down_threshold=down, horizon=HORIZON,
                 train_end=train_end, val_end=val_end, fast=fast, random_state=seed)
    return S.train_all(cfg, ohlcv)


@st.cache_resource(show_spinner=False, max_entries=64)
def get_backtest(_study, tkey, stage, fee, slip, rule, minconf, engine):
    return S.backtest_stage(_study, stage, fee, slip, rule, minconf, engine)


@st.cache_resource(show_spinner=False, max_entries=32)
def get_sweep(_study, tkey, stage, names, fees, slip, rule, minconf, engine):
    return S.fee_sweep(_study, stage, list(names), list(fees), slip, rule, minconf, engine)


@st.cache_data(show_spinner=False)
def eda_images(source, ticker, start, end, up, down, train_end, val_end):
    cfg, ohlcv, ds, sp = get_prepared(source, ticker, start, end, up, down, train_end, val_end)
    return {
        "price": fig_png(P.plot_price_with_splits(ohlcv, cfg)),
        "classes": fig_png(P.plot_class_distribution({"train": sp.y_train, "validation": sp.y_val, "test": sp.y_test})),
        "fwd": fig_png(P.plot_forward_return_hist(ds.fwd_ret, cfg)),
        "feat": fig_png(P.plot_feature_distributions(ds.X)),
        "corr": fig_png(P.plot_correlation(ds.X)),
        "cond": fig_png(P.plot_feature_vs_target(ds.X, ds.y)),
    }


# =============================================================================== sidebar
if "params" not in st.session_state:
    st.session_state["params"] = dict(DEFAULTS)
cur = st.session_state["params"]

st.sidebar.title("SPY Trading Signal Lab")
with st.sidebar.form("data_target_form"):
    st.subheader("1. Data & target (re-trains)")
    source = st.selectbox("Data source", ["auto", "yfinance", "csv", "synthetic"],
                          index=["auto", "yfinance", "csv", "synthetic"].index(cur["source"]),
                          help="auto = cached CSV, else yfinance. 'synthetic' is a smoke test only.")
    ticker = st.text_input("Ticker", value=cur["ticker"])
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=cur["start"], min_value=date(1993, 2, 1), max_value=date.today())
    end = c2.date_input("End", value=cur["end"], min_value=date(1993, 2, 1), max_value=date.today())
    st.caption(f"Target: Buy / Sell / Hold from the future **{HORIZON}-day** return.")
    up = st.slider("Buy if 5-day return >  (%)", 0.1, 20.0, float(cur["up"]), 0.1, format="%.1f%%")
    linked = st.checkbox("Use the same threshold for Sell", value=cur["linked"])
    down = st.slider("Sell if 5-day return < -(%)", 0.1, 20.0, float(cur["down"]), 0.1, format="%.1f%%",
                     help="Ignored when the checkbox above is ticked.")
    with st.expander("Split dates & training speed"):
        train_end = st.date_input("Train ends (inclusive)", value=cur["train_end"], min_value=date(1993, 2, 1), max_value=date.today())
        val_end = st.date_input("Validation ends (inclusive)", value=cur["val_end"], min_value=date(1993, 2, 1), max_value=date.today())
        fast = st.checkbox("Fast mode (small hyper-parameter grids)", value=cur["fast"])
    applied = st.form_submit_button("Apply & train")

if applied:
    st.session_state["params"] = dict(source=source, ticker=ticker.strip().upper() or "SPY", start=start, end=end,
                                      up=float(up), down=float(up if linked else down), linked=linked,
                                      train_end=train_end, val_end=val_end, fast=fast)
p = st.session_state["params"]

st.sidebar.subheader("2. Trading costs (instant)")
fee_pct = st.sidebar.slider("Transaction fee per trade (%)", 0.0, 2.0, 0.10, 0.01, format="%.2f%%")
slip_pct = st.sidebar.slider("Slippage per trade (%)", 0.0, 1.0, 0.05, 0.01, format="%.2f%%")
rule_label = st.sidebar.radio("Trading rule", ["Stateful (Buy=long, Sell=cash, Hold=keep)", "Daily Buy-only"], index=0)
rule = "stateful" if rule_label.startswith("Stateful") else "daily"
minconf = st.sidebar.slider("Ignore signals below confidence", 0.0, 0.9, 0.0, 0.05)
engine = st.sidebar.selectbox("Backtest engine", ["auto", "vectorbt", "builtin"], index=0,
                              help="auto = vectorbt if installed, else the built-in reference engine.")
fee, slip = fee_pct / 100, slip_pct / 100
up_f, down_f = p["up"] / 100, p["down"] / 100
dkey = (p["source"], p["ticker"], str(p["start"]), str(p["end"]), up_f, down_f, str(p["train_end"]), str(p["val_end"]))
tkey = dkey + (bool(p["fast"]), 42)

# =============================================================================== data + study
st.title("SPY Buy / Sell / Hold - Signal Lab")
if p["source"] == "synthetic":
    st.warning("SYNTHETIC DATA - for exercising the dashboard only. Nothing here says anything about the real market.")

try:
    with st.spinner("Loading data ..."):
        cfg0, ohlcv, ds, sp = get_prepared(*dkey)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not prepare the data: {exc}")
    st.stop()

study, study_msg = None, ""
try:
    with st.spinner("Tuning & fitting 2 baselines + 3 models (cached after the first run) ..."):
        study = get_study(*tkey)
except S.StudyError as exc:
    study_msg = str(exc)
except Exception as exc:  # noqa: BLE001
    study_msg = f"Training failed: {exc}"

if study is not None:
    best = study.runs[study.best_name]
    bt_hdr = get_backtest(study, tkey, "test", fee, slip, rule, minconf, engine)
    mr, br = bt_hdr["perf"].loc[study.best_name, "total_return"], bt_hdr["perf"].loc["Buy & Hold SPY", "total_return"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected model (validation macro-F1)", study.best_name)
    m2.metric("Test accuracy", f"{best.test_summary['accuracy']:.3f}",
              f"majority baseline {best.test_summary['majority_baseline_accuracy']:.3f}", delta_color="off")
    m3.metric("Selected model, test return", pct(mr), f"{(mr - br) * 100:+.1f} pts vs Buy & Hold")
    m4.metric("Cost per trade", f"{(fee + slip) * 100:.2f}%", f"fee {fee_pct:.2f}% + slippage {slip_pct:.2f}%", delta_color="off")
else:
    st.warning(study_msg)

tabs = st.tabs(["Overview", "1 EDA", "2 Modeling & explainability", "3 Backtest vs Buy & Hold",
                "4 All models: validation vs backtest", "5 Lessons learned"])


def need_study():
    if study is None:
        st.info("Model results are unavailable for these settings - see the warning at the top. "
                "The EDA tab still works so you can see why.")
        return True
    return False


# =============================================================================== Overview
with tabs[0]:
    st.subheader("Required deliverables -> where to find them")
    st.table(pd.DataFrame({
        "Deliverable": ["EDA (distributions, correlations, class balance)", "Modeling (pipeline, features, tuning)",
                        "Realistic backtest with fees + slippage", "Benchmark: model portfolio vs Buy & Hold",
                        "Lessons Learned report", "All models: validation vs backtest"],
        "Tab": ["1 EDA", "2 Modeling & explainability", "3 Backtest vs Buy & Hold", "3 Backtest vs Buy & Hold",
                "5 Lessons learned", "4 All models"]}))
    st.subheader("Current settings")
    st.write(f"**Buy** if {HORIZON}-day return > +{p['up']:.1f}% | **Sell** if < -{p['down']:.1f}% | otherwise **Hold**. "
             f"Train <= {p['train_end']}, validation <= {p['val_end']}, test afterwards "
             f"({HORIZON}-day purge at each boundary). Class weights = balanced. Tuning = purged TimeSeriesSplit.")
    st.write("Thresholds / splits need **Apply & train**; fees, slippage and the trading rule update instantly.")
    feas = S.class_feasibility(sp.y_train)
    st.write("Training-period class counts:", feas["counts"])
    if not feas["ok"]:
        st.error("Fewer than two classes have enough training examples at this threshold - lower it.")
    st.caption("Execution assumptions: signal at the close of day t, fill at the open of t+1, long-only, no leverage. "
               f"Backtest engine in use: {bt_hdr['bh'].engine if study is not None else 'n/a'}.")

# =============================================================================== EDA
with tabs[1]:
    imgs = eda_images(*dkey)
    st.subheader("Price and chronological splits")
    st.image(imgs["price"])
    st.subheader("Target class distribution at the chosen thresholds")
    a, b = st.columns([1, 2])
    with a:
        st.dataframe(class_distribution(ds.y).round(3))
        st.dataframe(sp.summary())
        st.caption(f"Purged rows: train {sp.purged_train}, validation {sp.purged_val}.")
    with b:
        st.image(imgs["classes"])
    st.image(imgs["fwd"])
    st.subheader("How the threshold changes class balance (symmetric +-x%)")
    rows = []
    fwd_train = ds.fwd_ret.loc[ds.fwd_ret.index <= pd.Timestamp(p["train_end"])]
    for t in sorted({0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, up_f}):
        lab = np.where(ds.fwd_ret > t, 1, np.where(ds.fwd_ret < -t, -1, 0))
        lab_tr = np.where(fwd_train > t, 1, np.where(fwd_train < -t, -1, 0))
        counts = {k: int((lab_tr == v).sum()) for k, v in (("Sell", -1), ("Hold", 0), ("Buy", 1))}
        rows.append({"threshold": f"+-{t:.1%}", "Sell": (lab == -1).mean(), "Hold": (lab == 0).mean(), "Buy": (lab == 1).mean(),
                     "trainable": "yes" if sum(v >= S.MIN_ROWS_PER_CLASS for v in counts.values()) >= 2 else "NO"})
    sens_tbl = pd.DataFrame(rows).set_index("threshold")
    st.dataframe(sens_tbl.assign(**{c: sens_tbl[c].map(lambda v: f"{v:.1%}") for c in ("Sell", "Hold", "Buy")}))
    st.caption("Wide thresholds make almost every day Hold; once fewer than two classes have enough training examples there is nothing to learn.")
    st.subheader("Feature distributions and correlations")
    st.image(imgs["feat"])
    st.image(imgs["corr"])
    corr = ds.X.corr().abs()
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False).head(8)
    st.write("Most correlated feature pairs (|r|):")
    st.dataframe(pairs.rename("abs_corr").round(3))
    st.subheader("Do features separate the classes?")
    st.image(imgs["cond"])
    st.caption("Feature groups: " + "; ".join(f"{g} ({len(v)})" for g, v in FEATURE_GROUPS.items()))

# =============================================================================== Modeling
with tabs[2]:
    if not need_study():
        st.subheader("Integrity audits")
        st.dataframe(study.audits)
        st.subheader("Validation-stage comparison (models tuned on TRAIN only)")
        val_tbl = S.classification_table(study, "validation")
        st.dataframe(val_tbl.round(3))
        if best.val_summary["majority_baseline_accuracy"] > 0.6:
            st.info("The majority class is very common at this threshold, so raw accuracy looks high for everyone. "
                    "Judge models by balanced accuracy / macro-F1 and the per-class recalls.")
        cv_tbl = pd.DataFrame({n: {"cv_macro_f1": r.cv_score, "best_params": str(r.best_params or ""), "note": r.note}
                               for n, r in study.runs.items() if r.kind == "model"}).T
        st.write("Hyper-parameter tuning (expanding-window `TimeSeriesSplit`, gap = horizon):")
        st.dataframe(cv_tbl)
        names = list(study.runs)
        m = st.selectbox("Model for confusion matrices", names, index=names.index(study.best_name))
        c1, c2 = st.columns(2)
        with c1:
            show(P.plot_confusion(study.runs[m].val_summary["confusion"], f"Validation - {m}"))
        with c2:
            show(P.plot_confusion(study.runs[m].test_summary["confusion"], f"Test - {m}"))
        st.subheader("Explainability")
        mm = st.selectbox("Model for permutation importance", study.model_names,
                          index=study.model_names.index(study.best_name))
        with st.spinner("Computing permutation importance on the validation window ..."):
            perm = S.permutation_for(study, mm)
        e1, e2 = st.columns(2)
        with e1:
            show(P.plot_importance(perm))
        with e2:
            st.dataframe(perm.head(12).round(4))
        if "logreg" in study.runs:
            lr = study.runs["logreg"].train_only_model
            st.write("**Logistic regression: signed coefficients per class** (standardised features)")
            show(P.plot_logreg_coefs(explain.logreg_coefficients(lr, ds.X.columns)))
            st.write("**Why did the logistic model say what it said on one day?**")
            n_val = len(sp.X_val)
            i = st.slider("Validation day #", 0, n_val - 1, n_val // 2)
            day = sp.X_val.index[i]
            tbl = explain.explain_prediction_logreg(lr, sp.X_val.loc[[day]])
            st.write(f"{day.date()}: predicts **{tbl.attrs['predicted']}** (actual: {evaluate.NAMES[int(sp.y_val.loc[day])]})")
            st.dataframe(tbl.round(4))
        if "random_forest" in study.runs:
            ti = explain.tree_importance(study.runs["random_forest"].train_only_model, ds.X.columns)
            if ti is not None:
                st.write("**Random-forest impurity importance** (train-fit; tends to favour continuous features)")
                st.bar_chart(ti.head(12))

# =============================================================================== Backtest
with tabs[3]:
    if not need_study():
        stage_label = st.radio("Window", ["Test (final backtest, untouched data)", "Validation (in-sample for model selection)"],
                               horizontal=True)
        stage = "test" if stage_label.startswith("Test") else "validation"
        bt = get_backtest(study, tkey, stage, fee, slip, rule, minconf, engine)
        names = list(study.runs)
        sel = st.multiselect("Strategies to plot", names, default=[study.best_name])
        st.caption(f"Window {bt['index'].min().date()} -> {bt['index'].max().date()} | fee {fee_pct:.2f}% + slippage {slip_pct:.2f}% per order | "
                   f"engine: {bt['bh'].engine} | rule: {rule}")
        show(P.plot_equity([bt["bh"], *[bt["results"][n] for n in sel]],
                           title=f"{stage.capitalize()} window: portfolio value after costs vs Buy & Hold"))
        st.dataframe(perf_display(bt["perf"]))
        st.download_button("Download performance CSV", bt["perf"].to_csv().encode(), f"backtest_{stage}.csv", "text/csv")
        st.subheader("Fee sensitivity: the same signals at commission levels from 0% to 2%")
        fees = tuple(sorted({0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, round(fee, 6)}))
        sweep = get_sweep(study, tkey, stage, tuple(sel), fees, slip, rule, minconf, engine)
        metric = st.radio("Metric", ["total_return", "sharpe", "n_orders"], horizontal=True)
        show(P.plot_fee_sweep(sweep, metric, f"{metric.replace('_', ' ')} vs commission ({stage})", pct=(metric == "total_return")))
        piv = sweep.pivot(index="fee", columns="strategy", values=metric)
        piv.index = [f"{f:.2%}" for f in piv.index]
        st.dataframe(piv.round(3))

# =============================================================================== All models
with tabs[4]:
    if not need_study():
        val_tbl = S.classification_table(study, "validation")
        test_tbl = S.classification_table(study, "test")
        val_bt = get_backtest(study, tkey, "validation", fee, slip, rule, minconf, engine)
        test_bt = get_backtest(study, tkey, "test", fee, slip, rule, minconf, engine)
        names = list(study.runs)

        st.subheader("Scorecard: every model, validation stage vs backtest (test) stage")
        score = pd.DataFrame({
            "kind": val_tbl["kind"],
            "val_macro_f1": val_tbl["macro_f1"], "test_macro_f1": test_tbl["macro_f1"],
            "val_bal_acc": val_tbl["balanced_accuracy"], "test_bal_acc": test_tbl["balanced_accuracy"],
            "val_return": val_bt["perf"]["total_return"].reindex(names), "test_return": test_bt["perf"]["total_return"].reindex(names),
            "test_sharpe": test_bt["perf"]["sharpe"].reindex(names), "test_max_dd": test_bt["perf"]["max_drawdown"].reindex(names),
            "test_orders": test_bt["perf"]["n_orders"].reindex(names),
        })
        score["test_vs_buy_hold"] = score["test_return"] - test_bt["perf"].loc["Buy & Hold SPY", "total_return"]
        st.dataframe(score.round(3))
        st.download_button("Download scorecard CSV", score.to_csv().encode(), "scorecard.csv", "text/csv")

        g1, g2 = st.columns(2)
        with g1:
            show(P.plot_metric_bars(score[["val_macro_f1", "test_macro_f1"]].rename(columns={"val_macro_f1": "Validation", "test_macro_f1": "Test"}),
                                    "Macro-F1: validation vs test", "macro-F1"))
        with g2:
            ret = score[["val_return", "test_return"]].rename(columns={"val_return": "Validation window", "test_return": "Test window"})
            ret.loc["Buy & Hold SPY"] = [val_bt["perf"].loc["Buy & Hold SPY", "total_return"], test_bt["perf"].loc["Buy & Hold SPY", "total_return"]]
            show(P.plot_metric_bars(ret, f"Total return after costs ({fee_pct:.2f}% fee)", "total return", pct=True))

        st.subheader("Validation stage")
        st.write("Classification metrics (train-only models, validation window):")
        st.dataframe(val_tbl.round(3))
        st.write("Strategy performance on the validation window (in-sample for model selection):")
        st.dataframe(perf_display(val_bt["perf"]))
        show(P.plot_equity([val_bt["bh"], *val_bt["results"].values()], title="Validation window: all strategies vs Buy & Hold"))

        st.subheader("Backtest stage (test window, models re-fit on train + validation)")
        st.write("Classification metrics (test window, scored once):")
        st.dataframe(test_tbl.round(3))
        st.write("Strategy performance on the test window:")
        st.dataframe(perf_display(test_bt["perf"]))
        show(P.plot_equity([test_bt["bh"], *test_bt["results"].values()], title="Test window: all strategies vs Buy & Hold"))
        st.caption("A model that ranks well on validation but poorly on test (or vice-versa) is the normal fingerprint of non-stationary, noisy data.")

# =============================================================================== Lessons
with tabs[5]:
    if not need_study():
        with st.spinner("Building report ..."):
            res = S.report_inputs(study, fee, slip, engine, minconf)
            note = "SYNTHETIC DATA - smoke test only; these numbers say nothing about the real market." if p["source"] == "synthetic" else ""
            report_md = build_report(res["cfg"], res, note)
        st.markdown(report_md)
        st.download_button("Download report (.md)", report_md.encode(), "lessons_learned.md", "text/markdown")
        st.subheader("Your own analysis (the grade is for interpretation)")
        mine = st.text_area("Write your lessons learned here", height=220,
                            placeholder="Where did the model fail? Why did it under/over-perform Buy & Hold? What do the fee curves say? ...")
        if mine.strip():
            st.download_button("Download report + my analysis", (report_md + "\n\n## My analysis\n\n" + mine).encode(),
                               "lessons_learned_final.md", "text/markdown")
