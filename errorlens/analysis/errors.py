"""Error identification: turn (y, y_pred) into discovery targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from errorlens.discovery.subgroup import DiscoveryTarget

SEVERE_QUANTILE = 0.90


def classification_targets(y: np.ndarray, y_pred: np.ndarray, labels: list[Any], *,
                           positive_label: Any, directional: bool, max_classes: int,
                           min_samples: int) -> list[DiscoveryTarget]:
    """Targets for classification.

    * ``error``: misclassification, population = all rows.
    * binary: ``false_negative`` (population = actual positives, metric = FNR) and
      ``false_positive`` (population = actual negatives, metric = FPR).
    * multiclass: ``class_error`` per class c (population = rows with y == c, metric = the
      class's miss rate, 1 - recall), for up to ``max_classes`` classes with the most misses.
    """
    n = len(y)
    errors = (y != y_pred).astype(float)
    targets = [DiscoveryTarget("error", errors, np.ones(n, dtype=bool), True, "error rate")]
    if not directional:
        return targets
    if len(labels) == 2:
        pos = y == positive_label
        neg = ~pos
        targets.append(DiscoveryTarget(
            "false_negative", (y_pred != positive_label).astype(float), pos, True,
            "false negative rate", target_label=positive_label))
        targets.append(DiscoveryTarget(
            "false_positive", (y_pred == positive_label).astype(float), neg, True,
            "false positive rate", target_label=positive_label))
    elif len(labels) > 2:
        misses = []
        for label in labels:
            members = y == label
            n_missed = int((members & (y_pred != label)).sum())
            if members.sum() >= 2 * min_samples and n_missed >= 10:
                misses.append((n_missed, label))
        for _, label in sorted(misses, key=lambda v: -v[0])[:max_classes]:
            targets.append(DiscoveryTarget(
                "class_error", (y_pred != label).astype(float), y == label, True,
                f"miss rate for class {label}", target_label=label))
    return targets


def regression_targets(y: np.ndarray, y_pred: np.ndarray, *,
                       directional: bool) -> tuple[list[DiscoveryTarget], float]:
    """Targets for regression.

    * ``high_error``: continuous absolute error (subgroup metric = MAE).
    * ``underprediction``: residual > tau, ``overprediction``: residual < -tau, where
      tau is the 90th percentile of the absolute residual (severe errors).

    Returns the targets and tau.
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y - y_pred
    abs_err = np.abs(residual)
    n = len(y)
    everyone = np.ones(n, dtype=bool)
    tau = float(np.quantile(abs_err, SEVERE_QUANTILE)) if n else 0.0
    targets = [DiscoveryTarget("high_error", abs_err, everyone, False, "MAE",
                               residuals=residual)]
    if directional and tau > 0:
        targets.append(DiscoveryTarget("underprediction", (residual > tau).astype(float),
                                       everyone, True, "severe underprediction rate",
                                       residuals=residual))
        targets.append(DiscoveryTarget("overprediction", (residual < -tau).astype(float),
                                       everyone, True, "severe overprediction rate",
                                       residuals=residual))
    return targets, tau
