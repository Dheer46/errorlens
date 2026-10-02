"""Classification performance metrics and error summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skm

from errorlens.core.types import (
    ClassificationPerformance,
    ClassMetrics,
    ConfusionPair,
    ErrorSummary,
)

HIGH_CONFIDENCE = 0.9


def default_positive_label(labels: list[Any]) -> Any:
    """``True`` or ``1`` when present, otherwise the last label in sorted order."""
    for label in labels:
        if isinstance(label, (bool, np.bool_)) and bool(label):
            return label
    for label in labels:
        if isinstance(label, (int, float, np.integer, np.floating)) and label == 1:
            return label
    return labels[-1]


def classification_performance(y: np.ndarray, y_pred: np.ndarray, labels: list[Any], *,
                               positive_label: Any = None, proba: np.ndarray | None = None,
                               proba_classes: list[Any] | None = None
                               ) -> ClassificationPerformance:
    n = len(y)
    correct = y == y_pred
    n_errors = int((~correct).sum())
    is_binary = len(labels) == 2
    prec, rec, f1, support = skm.precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0)
    cm = skm.confusion_matrix(y, y_pred, labels=labels)
    per_class = []
    for i, label in enumerate(labels):
        per_class.append(ClassMetrics(
            label=label, precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]),
            support=int(support[i]), n_missed=int(cm[i].sum() - cm[i, i]),
            n_false_alarms=int(cm[:, i].sum() - cm[i, i]),
        ))
    present = [i for i, c in enumerate(per_class) if c.support > 0]
    perf = ClassificationPerformance(
        n_samples=n,
        n_errors=n_errors,
        accuracy=float(correct.mean()),
        error_rate=n_errors / n,
        balanced_accuracy=float(np.mean([rec[i] for i in present])) if present else float("nan"),
        macro_precision=float(np.mean(prec)),
        macro_recall=float(np.mean([rec[i] for i in present])) if present else float("nan"),
        macro_f1=float(np.mean(f1)),
        labels=list(labels),
        confusion_matrix=cm.astype(int).tolist(),
        per_class=per_class,
        is_binary=is_binary,
    )
    if is_binary:
        pos = positive_label if positive_label is not None else default_positive_label(labels)
        perf.positive_label = pos
        y_pos, p_pos = y == pos, y_pred == pos
        perf.tp = int((y_pos & p_pos).sum())
        perf.tn = int((~y_pos & ~p_pos).sum())
        perf.fp = int((~y_pos & p_pos).sum())
        perf.fn = int((y_pos & ~p_pos).sum())
        n_pos, n_neg = perf.tp + perf.fn, perf.tn + perf.fp
        perf.false_negative_rate = perf.fn / n_pos if n_pos else float("nan")
        perf.false_positive_rate = perf.fp / n_neg if n_neg else float("nan")
    if proba is not None and proba_classes is not None:
        _add_probability_metrics(perf, y, proba, proba_classes)
    return perf


def _add_probability_metrics(perf: ClassificationPerformance, y: np.ndarray, proba: np.ndarray,
                             proba_classes: list[Any]) -> None:
    if len(np.unique(y)) < 2 or proba.shape[1] != len(proba_classes):
        return
    try:
        if perf.is_binary and perf.positive_label in proba_classes:
            col = proba_classes.index(perf.positive_label)
            perf.roc_auc = float(skm.roc_auc_score(y == perf.positive_label, proba[:, col]))
        elif not perf.is_binary and set(np.unique(y)) <= set(proba_classes):
            perf.roc_auc = float(skm.roc_auc_score(y, proba, multi_class="ovr",
                                                   labels=proba_classes))
        if set(np.unique(y)) <= set(proba_classes):
            perf.log_loss = float(skm.log_loss(y, np.clip(proba, 1e-15, 1),
                                               labels=proba_classes))
    except ValueError:
        # Probability metrics are optional context; inconsistent label sets leave them unset.
        pass


def classification_error_summary(y: np.ndarray, y_pred: np.ndarray,
                                 proba: np.ndarray | None = None,
                                 max_pairs: int = 10) -> ErrorSummary:
    errors = y != y_pred
    n_errors = int(errors.sum())
    summary = ErrorSummary(n_errors=n_errors, error_rate=n_errors / len(y))
    if n_errors:
        pairs: dict[tuple[Any, Any], int] = {}
        for t, p in zip(y[errors].tolist(), y_pred[errors].tolist()):
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:max_pairs]
        summary.top_confusions = [ConfusionPair(t, p, c, c / n_errors) for (t, p), c in top]
    if proba is not None:
        confidence = proba.max(axis=1)
        if (~errors).any():
            summary.mean_confidence_correct = float(confidence[~errors].mean())
        if errors.any():
            summary.mean_confidence_errors = float(confidence[errors].mean())
        summary.high_confidence_threshold = HIGH_CONFIDENCE
        summary.n_high_confidence_errors = int((errors & (confidence >= HIGH_CONFIDENCE)).sum())
    return summary
