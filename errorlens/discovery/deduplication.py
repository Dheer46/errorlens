"""Redundancy control: condition simplification and overlap-aware de-duplication."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from errorlens.analysis.statistics import fisher_greater, welch_greater
from errorlens.core.types import Condition

T = TypeVar("T")

SIMPLIFY_TOLERANCE = 0.10  # a condition is dropped if removing it lowers the metric < 10 %
REFINEMENT_GAIN = 0.10      # a sub-region must raise the metric by >= 10 % to be kept
RESIDUAL_EXCESS_SHARE = 0.5  # the non-overlapping part must keep >= 50 % of the excess
REFINEMENT_ALPHA = 0.05     # a refinement must be detectably worse than its parent


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return inter / union


def simplify(conditions: tuple[Condition, ...], mask_of: Callable[[tuple[Condition, ...]],
                                                                    np.ndarray],
             target: np.ndarray, min_support: int,
             tolerance: float = SIMPLIFY_TOLERANCE) -> tuple[Condition, ...]:
    """Greedily drop conditions that do not materially raise the subgroup's error metric.

    Produces minimal descriptions: ``age < 25 AND region == north`` becomes ``age < 25`` if
    restricting to ``region == north`` adds (almost) nothing.
    """
    current = tuple(conditions)
    while len(current) > 1:
        mask = mask_of(current)
        n = int(mask.sum())
        if n == 0:
            return current
        value = float(target[mask].mean())
        best: tuple[float, tuple[Condition, ...]] | None = None
        for i in range(len(current)):
            reduced = current[:i] + current[i + 1:]
            rmask = mask_of(reduced)
            if int(rmask.sum()) < min_support:
                continue
            rvalue = float(target[rmask].mean())
            if rvalue >= value * (1 - tolerance) and (best is None or rvalue > best[0]):
                best = (rvalue, reduced)
        if best is None:
            break
        current = best[1]
    return current


def is_meaningful_refinement(mask: np.ndarray, parent: np.ndarray, target: np.ndarray,
                             alpha: float = REFINEMENT_ALPHA) -> bool:
    """Whether a sub-region of ``parent`` is materially *and* detectably worse than the rest.

    Requires the sub-region's metric to be >= 10 % higher than the parent's, and a one-sided
    test of sub-region vs. the remainder of the parent to reach ``alpha``. This is a
    heuristic redundancy filter on the data at hand, not a reported inference.
    """
    inside = mask & parent
    remainder = parent & ~mask
    if not inside.any():
        return False
    value = float(target[inside].mean())
    parent_value = float(target[parent].mean())
    if not value >= parent_value * (1 + REFINEMENT_GAIN):
        return False
    if int(remainder.sum()) < 2:
        return False
    a, b = target[inside], target[remainder]
    if np.isin(target, (0.0, 1.0)).all():
        p = fisher_greater(int(a.sum()), len(a), int(a.sum() + b.sum()), len(a) + len(b))
    else:
        p = welch_greater(a, b)
    return p < alpha


def is_redundant(mask: np.ndarray, other: np.ndarray, target: np.ndarray, baseline: float,
                 min_support: int, jaccard_threshold: float,
                 refines: Callable[[], bool] | None = None) -> bool:
    """Whether subgroup ``mask`` adds nothing beyond the (better-ranked) subgroup ``other``.

    A subgroup is redundant if

    1. its rows overlap ``other`` with Jaccard similarity >= ``jaccard_threshold``; or
    2. it is (almost) a sub-region of ``other`` -- fewer than ``min_support`` rows lie outside
       ``other`` -- and it is not a meaningful refinement (``refines``, by default
       :func:`is_meaningful_refinement` on the same data); or
    3. the rows it does *not* share with ``other`` retain less than half of its excess error
       over the baseline, i.e. its elevation is explained by the overlap (a "diluted"
       version of ``other``).
    """
    if jaccard(mask, other) >= jaccard_threshold:
        return True
    inter = mask & other
    if not inter.any():
        return False
    n = int(mask.sum())
    value = float(target[mask].mean())
    rest = mask & ~other
    n_rest = int(rest.sum())
    if n_rest < min_support:
        ok = refines() if refines is not None else is_meaningful_refinement(mask, other, target)
        return not ok
    excess = value - baseline
    if excess <= 0 or n == 0:
        return True
    rest_excess = float(target[rest].mean()) - baseline
    return rest_excess < RESIDUAL_EXCESS_SHARE * excess


def deduplicate(items: Sequence[T], *, conditions_of: Callable[[T], tuple[Condition, ...]],
                mask_of: Callable[[T], np.ndarray], target: np.ndarray, min_support: int,
                threshold: float, refines: Callable[[T, T], bool] | None = None) -> list[T]:
    """Greedy de-duplication of items already sorted best-first.

    An item is dropped if it refines a kept item (strict superset of conditions, or a
    sub-region of its rows) without being a meaningful refinement, or if
    :func:`is_redundant` says a kept item explains it. ``refines(child, parent)`` overrides
    the refinement test (e.g. to evaluate it on independent validation data).
    """
    baseline = float(np.mean(target)) if len(target) else 0.0
    kept: list[T] = []
    kept_masks: list[np.ndarray] = []
    for item in items:
        mask = mask_of(item)
        conds = set(conditions_of(item))
        redundant = False
        for other, omask in zip(kept, kept_masks):

            def check(item: T = item, other: T = other, mask: np.ndarray = mask,
                      omask: np.ndarray = omask) -> bool:
                if refines is not None:
                    return refines(item, other)
                return is_meaningful_refinement(mask, omask, target)

            if set(conditions_of(other)) < conds and not check():
                redundant = True
                break
            if is_redundant(mask, omask, target, baseline, min_support, threshold, check):
                redundant = True
                break
        if not redundant:
            kept.append(item)
            kept_masks.append(mask)
    return kept
