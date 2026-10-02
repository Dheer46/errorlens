"""Scoring and ranking of validated failure patterns."""

from __future__ import annotations

import math
from collections.abc import Callable

from errorlens.core.types import FailurePattern

COMPLEXITY_PENALTY = 0.25


def excess_lower_bound(pattern: FailurePattern) -> float:
    """Conservative excess of the error metric in the subgroup, scaled to its full size.

    ``n_full * (CI_low - baseline)``: for binary targets, a lower bound on how many more
    errors the subgroup contains than the baseline rate would predict; for continuous
    targets the same quantity in units of absolute error.
    """
    v = pattern.validation
    if math.isnan(v.ci_low) or math.isnan(v.baseline):
        return 0.0
    return pattern.stats.n * (v.ci_low - v.baseline)


def pattern_score(pattern: FailurePattern) -> float:
    """Default ranking score: conservative excess errors with a complexity penalty."""
    return excess_lower_bound(pattern) / (1.0 + COMPLEXITY_PENALTY * (pattern.complexity - 1))


def sort_key(rank_by: str) -> Callable[[FailurePattern], tuple[float, float]]:
    """Return a key function sorting patterns best-first for the given criterion."""

    def key(p: FailurePattern) -> tuple[float, float]:
        tie = -p.score
        if rank_by == "lift":
            return (-_finite(p.lift), tie)
        if rank_by == "p_value":
            return (_finite(p.p_value_adjusted, 1.0), tie)
        if rank_by == "coverage":
            return (-_finite(p.coverage), tie)
        if rank_by == "error_coverage":
            return (-_finite(p.stats.event_coverage), tie)
        return (-p.score, _finite(p.p_value_adjusted, 1.0))

    return key


def _finite(value: float, default: float = 0.0) -> float:
    if value is None or math.isnan(value):
        return default
    return value
