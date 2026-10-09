"""Interpretable tabular classifiers + time-aware hyper-parameter tuning.

Order of complexity follows the assignment: Logistic Regression -> Random Forest
-> Gradient Boosting. Class imbalance is handled with class weights
(`balanced`), which re-weights the loss so Sell/Hold/Buy matter equally.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def model_zoo(cfg: Config) -> dict:
    """name -> (pipeline, param_grid)."""
    rs = cfg.random_state
    lr = Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=rs))])
    rf = Pipeline([("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                                  random_state=rs))])
    hgb = Pipeline([("clf", HistGradientBoostingClassifier(class_weight="balanced", l2_regularization=1.0,
                                                           random_state=rs))])
    if cfg.fast:
        return {
            "logreg": (lr, {"clf__C": [0.1, 1.0]}),
            "random_forest": (rf, {"clf__n_estimators": [100], "clf__max_depth": [3, 5],
                                   "clf__min_samples_leaf": [50]}),
            "grad_boosting": (hgb, {"clf__max_depth": [2], "clf__learning_rate": [0.05], "clf__max_iter": [100]}),
        }
    return {
        "logreg": (lr, {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (rf, {"clf__n_estimators": [300], "clf__max_depth": [3, 5, 8],
                               "clf__min_samples_leaf": [20, 50, 100]}),
        "grad_boosting": (hgb, {"clf__max_depth": [2, 3], "clf__learning_rate": [0.03, 0.1],
                                "clf__max_iter": [100, 200], "clf__min_samples_leaf": [20, 50]}),
    }


def baseline_models() -> dict:
    """The bar every real model must clear (same lesson as the L06 baseline rule)."""
    return {"baseline_majority": DummyClassifier(strategy="most_frequent"),
            "baseline_stratified_random": DummyClassifier(strategy="stratified", random_state=0)}


def time_series_cv(cfg: Config) -> TimeSeriesSplit:
    # gap=horizon purges the overlap between each CV train fold and its validation fold
    return TimeSeriesSplit(n_splits=cfg.n_cv_splits, gap=cfg.horizon)


def tune(name: str, estimator, grid: dict, X: pd.DataFrame, y: pd.Series, cfg: Config) -> GridSearchCV:
    """Grid search with expanding-window, purged TimeSeriesSplit - never KFold/shuffle."""
    search = GridSearchCV(estimator, grid, cv=time_series_cv(cfg), scoring=cfg.scoring, n_jobs=1,
                          refit=True, return_train_score=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search
