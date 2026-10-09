# S&P 500 (SPY) Buy / Sell / Hold - Explainable ML Trading Signal Model

Time-series classification project built to the "Machine Learning Project: S&P 500 Trading Model" brief.
**Goal: methodology, not profit** - build, validate and backtest a model on noisy market data *without data leakage*,
and understand why good classification does not automatically mean a good trading strategy.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py                  # real SPY via yfinance -> figures, tables, lessons_learned.md in ./outputs
jupyter notebook notebooks/             # 01_eda -> 02_modeling -> 03_backtest_and_lessons
python -m unittest discover -s tests -v # 20 tests incl. leakage, backtest accounting, multi-model study, dashboard
```

### Interactive dashboard

```bash
streamlit run streamlit_app.py
```

| Sidebar control | Range | Effect |
|---|---|---|
| Buy / Sell threshold on the future **5-day return** | 0.1% - **20%** (optionally linked) | re-labels the data and re-tunes all models (click **Apply & train**; cached afterwards) |
| Transaction fee per trade | 0 - **2%** | instant: only the backtest is recomputed, models are not retrained |
| Slippage, trading rule (stateful / daily Buy-only), confidence filter, engine | - | instant |
| Data source, dates, split dates, fast mode | - | part of the Apply & train form |

Tabs: **Overview** (deliverable map) - **1 EDA** (class balance, forward-return histogram, threshold-vs-balance table, feature distributions, correlations)
- **2 Modeling & explainability** (audits, validation comparison, CV tuning, confusion matrices, permutation importance, logistic coefficients, per-day explanation)
- **3 Backtest vs Buy & Hold** (test or validation window, selectable strategies, fee sensitivity 0-2%)
- **4 All models: validation vs backtest** (scorecard, side-by-side charts, both stages for every model and baseline)
- **5 Lessons learned** (auto-generated report from the current settings, plus a box for your own analysis, downloadable).

Behaviour at extreme settings: when a wide threshold (e.g. 20%) leaves fewer than two classes with enough training examples, the app says so
and keeps the EDA tab working instead of crashing; if only one of Buy/Sell disappears the models train as 2-class problems.
All dashboard logic lives in `src/spy_signals/study.py` (no Streamlit imports) so it is unit-tested.

Offline smoke test (synthetic data, **never interpret its numbers**): `python run_pipeline.py --source synthetic --fast`
If yfinance is blocked, put a CSV with columns `Date,Open,High,Low,Close,Volume` at `data/SPY_ohlcv.csv`.

## Deliverables -> where they live

| Assignment requirement | Where |
|---|---|
| EDA notebook (feature distributions, correlations, class distribution) | `notebooks/01_eda.ipynb` |
| Modeling notebook (pipeline, features, training, tuning) | `notebooks/02_modeling.ipynb` |
| Realistic backtest with a backtesting library + fees/slippage | `notebooks/03_backtest_and_lessons.ipynb`, `src/spy_signals/backtest.py` (vectorbt; 0.10% fee + 0.05% slippage per order) |
| Model vs Buy & Hold plot | `outputs/figures/08_equity_vs_buy_hold.png` (and notebook 03) |
| "Lessons Learned" report | `outputs/lessons_learned.md` (auto-filled from your results) + your own section at the end of notebook 03 |
| Returns, >= 3 indicators, volatility, strictly lagged features | `src/spy_signals/features.py` (23 features: returns, SMA ratios, RSI, MACD, Bollinger, rolling vol, ATR ...) |
| Buy/Sell/Hold thresholds on 5-day forward return | `src/spy_signals/target.py` (+-1%, configurable) |
| Chronological split / TimeSeriesSplit, no random split | `splits.py`, `models.py` (`TimeSeriesSplit(gap=horizon)`) |
| Class imbalance strategy | `class_weight="balanced"` in every model; threshold sensitivity table in the EDA notebook |
| Interpretable models first | Logistic Regression -> Random Forest -> Gradient Boosting (no deep learning) |
| Look-ahead audit | `audit.py` + `tests/` |

## Design decisions (the "why")

1. **Feature timing.** The row for date *t* uses only OHLCV through the close of *t*; its label is the return from close *t* to close *t+5*.
   The backtest then fills the order at the **open of t+1**, so execution is not magically at the price the signal was computed from.
2. **Purging.** 5-day labels reach into the next split, so the last 5 rows of train and validation are dropped, and CV uses `gap=5`.
   Splits default to Train <= 2020, Validation 2021-2022, Test 2023+ (edit `Config`).
3. **Imbalance.** Class-weighted losses + metrics that do not reward the majority class (balanced accuracy, macro-F1), compared with
   a majority baseline and a stratified-random baseline.
4. **Model selection discipline.** Tune on train (CV), select on validation, refit on train+validation, score test **once**.
5. **Long-only, stateful trading rule.** Buy -> long, Sell -> cash, Hold -> keep position. A "daily Buy-only" variant is included on purpose to show the cost of over-trading;
   a fee-sensitivity table re-runs everything at 0 / 0.05 / 0.1 / 0.2 / 0.5% commission.

## Explainability toolkit (`explain.py`)

* Global: permutation importance on held-out data (model-agnostic), logistic-regression coefficients per class, tree importances.
* Local: `explain_prediction_logreg` decomposes one day's logistic score into per-feature contributions (coefficient x standardised value).
* Caveat: correlated features (3 volatility windows) split credit, so each looks weaker than the group really is.

## Leakage audits (run automatically, reported in `outputs/audit.csv` and the report)

1. **Truncation test** - features at date *t* must be identical when computed on data truncated at *t*. (A planted-leak unit test proves this check fails when a feature peeks ahead.)
2. Target not among features; no suspicious feature/future-return correlation.
3. Train < Validation < Test with a gap larger than the label horizon.
4. Every CV fold trains strictly before it validates.
5. Preprocessing fitted inside the sklearn `Pipeline` (train only).
6. Plausibility alarm: test accuracy above 60% is flagged (assignment: ~35-40% is normal; 80% means a leak).

## Project layout

```
run_pipeline.py            one-command end-to-end run (CLI flags: --source --fast --engine --fee --horizon ...)
streamlit_app.py           interactive dashboard (thresholds up to 20%, fees up to 2%, all models x validation/backtest)
src/spy_signals/           config, data, features, target, splits, models, evaluate, explain, audit, backtest, pipeline, study, plotting, report, synthetic
notebooks/                 01_eda, 02_modeling, 03_backtest_and_lessons
tests/                     unit + leakage + backtest-accounting + end-to-end + study + dashboard tests
outputs/                   created on run (figures, CSVs, metrics.json, lessons_learned.md, final_model.joblib)
```

## Customising

Everything is in `Config` (`src/spy_signals/config.py`) or CLI flags: tickers, dates, horizon, thresholds, split dates, fee/slippage,
`min_confidence` (ignore low-confidence signals), CV folds. To add XGBoost/LightGBM, add an entry to `model_zoo()` in `models.py`.

## Verification status (please read)

* Developed and verified in a sandbox **without internet access**, so the code was exercised on **synthetic** SPY-like data only:
  12 unit tests pass, all three notebooks' code cells execute end to end, and all six audits pass. No real-SPY results were produced or inspected.
* `vectorbt` and `yfinance` were **not installable in that sandbox**, so the vectorbt code path (`backtest._vectorbt`) has not been run by me.
  It uses the standard `Portfolio.from_signals(... price=open, fees=, slippage=)` API with the same timing/cost conventions as the built-in reference engine, which *is* unit-tested
  against hand-computed values. Notebook 03 prints a vectorbt-vs-built-in cross-check on your machine; if they differ materially, investigate before trusting either.
* The built-in engine is a fallback for tests/cross-checks. To satisfy the "dedicated backtesting library" requirement, run with vectorbt installed (`--engine vectorbt` fails loudly if it is missing).
* Developed on Python 3.12 / pandas 3.0 / scikit-learn 1.8. `requirements.txt` pins pandas<2.3 and numpy<2.1 for vectorbt compatibility; that pinned combination was not tested here.
* **The Streamlit app has not been run in a real browser/Streamlit server** (Streamlit was not installable in the sandbox). Its script was executed
  against a fake `streamlit` module (`tests/test_streamlit_app.py`) and the underlying logic is tested directly, but Streamlit-specific behaviour
  (widget layout, form handling, caching) is untested. If a widget argument is rejected by your Streamlit version, the fix will be a one-line change in `streamlit_app.py`.
* The accuracy-plausibility audit alarms at max(0.60, majority-class accuracy + 0.15), because with wide thresholds Hold dominates and high accuracy can be legitimate.
* Expect (and accept) accuracy around 35-40% and underperformance vs Buy & Hold after costs - the project is graded on methodology and interpretation.
