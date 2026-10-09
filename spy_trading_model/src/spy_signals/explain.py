"""Model explainability: global (coefficients, permutation importance) and local."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluate import NAMES


def permutation_table(model, X: pd.DataFrame, y: pd.Series, scoring: str = "f1_macro",
                      n_repeats: int = 20, seed: int = 42) -> pd.DataFrame:
    """How much does shuffling ONE feature (on held-out data) hurt the score?

    Model-agnostic and measured on data the model was not fit on, so it reflects
    what the model actually relies on out-of-sample. Caveat: correlated features
    (e.g. vol_10d / vol_21d) share credit, so each looks less important alone.
    """
    res = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    out = pd.DataFrame({"importance_mean": res.importances_mean, "importance_std": res.importances_std},
                       index=X.columns)
    return out.sort_values("importance_mean", ascending=False)


def _class_coefs(clf):
    """(n_classes, n_features) coefficients + intercepts. sklearn stores ONE row for a binary
    problem (log-odds of classes_[1] vs classes_[0]); expand it symmetrically so 2- and 3-class
    cases are handled identically (needed when extreme thresholds make a class vanish)."""
    coef, icpt = np.asarray(clf.coef_), np.asarray(clf.intercept_)
    if coef.shape[0] == 1 and len(clf.classes_) == 2:
        coef, icpt = np.vstack([-coef[0], coef[0]]) / 2, np.array([-icpt[0], icpt[0]]) / 2
    return coef, icpt


def logreg_coefficients(pipe, feature_names) -> pd.DataFrame:
    """Per-class coefficients on STANDARDISED features (so magnitudes are comparable)."""
    clf = pipe.named_steps["clf"]
    coef, _ = _class_coefs(clf)
    return pd.DataFrame(coef.T, index=feature_names, columns=[NAMES[c] for c in clf.classes_])


def tree_importance(pipe, feature_names) -> pd.Series | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    return None


def explain_prediction_logreg(pipe, x_row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Local explanation for ONE day: contribution_j = coef_j(class) * standardised_x_j.

    The class score (logit) is intercept + sum of these contributions, so the table
    is an exact decomposition of why the model leaned toward its predicted class.
    """
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    z = scaler.transform(x_row)[0]
    pred = pipe.predict(x_row)[0]
    k = list(clf.classes_).index(pred)
    coef, icpt = _class_coefs(clf)
    contrib = pd.Series(coef[k] * z, index=x_row.columns)
    tbl = pd.DataFrame({"value": x_row.iloc[0], "z_score": z, "contribution": contrib})
    tbl = tbl.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
    tbl.attrs["predicted"] = NAMES[pred]
    tbl.attrs["intercept"] = float(icpt[k])
    return tbl
