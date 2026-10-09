"""Classification metrics with the baselines needed to read them honestly."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

LABELS = [-1, 0, 1]
NAMES = {-1: "Sell", 0: "Hold", 1: "Buy"}


def classification_summary(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "per_class": {NAMES[k]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                 "support": int(s[i])} for i, k in enumerate(LABELS)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def majority_accuracy(y_train, y_eval) -> float:
    """Accuracy of always predicting the most common TRAINING class."""
    top = pd.Series(y_train).value_counts().idxmax()
    return float((np.asarray(y_eval) == top).mean())


def comparison_table(rows: dict) -> pd.DataFrame:
    """rows: name -> classification_summary dict."""
    return pd.DataFrame({k: {m: v[m] for m in ("accuracy", "balanced_accuracy", "macro_f1")}
                         for k, v in rows.items()}).T.round(4)
