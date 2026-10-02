"""Statistical primitives: tests, confidence intervals, effect sizes and p-value adjustment.

All tests comparing a subgroup with its complement are **one-sided** (alternative: the
subgroup's error metric is *greater*), because ErrorLens only reports subgroups where the
model is worse than elsewhere.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from errorlens.core.types import GroupStats

Z_975 = 1.959963984540054


def is_effectively_constant(values: np.ndarray) -> bool:
    """True if ``values`` vary only at floating-point noise level."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(np.std(values)) <= 1e-9 * scale


# --------------------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------------------


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(sps.norm.ppf(0.5 + confidence / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_interval_vec(k: np.ndarray, n: np.ndarray, z: float = Z_975) -> tuple[np.ndarray,
                                                                                np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def mean_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Student-t confidence interval for a mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    m = float(values.mean())
    se = float(values.std(ddof=1)) / math.sqrt(n)
    if se == 0:
        return (m, m)
    t = float(sps.t.ppf(0.5 + confidence / 2, df=n - 1))
    return (m - t * se, m + t * se)


def relative_risk_interval(k1: int, n1: int, k0: int, n0: int,
                           confidence: float = 0.95) -> tuple[float, float, float]:
    """Relative risk (group 1 vs group 0) with a Katz log interval.

    A Haldane–Anscombe 0.5 correction is applied when any cell is zero.
    """
    if n1 <= 0 or n0 <= 0:
        return (float("nan"), float("nan"), float("nan"))
    a, b, c, d = float(k1), float(n1 - k1), float(k0), float(n0 - k0)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    r1, r0 = a / (a + b), c / (c + d)
    rr = r1 / r0
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    z = float(sps.norm.ppf(0.5 + confidence / 2))
    return (rr, math.exp(math.log(rr) - z * se), math.exp(math.log(rr) + z * se))


# --------------------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions (positive when p1 > p2)."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation (positive when mean(a) > mean(b))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled <= 0:
        return 0.0 if a.mean() == b.mean() else float("inf") * np.sign(a.mean() - b.mean())
    return float((a.mean() - b.mean()) / math.sqrt(pooled))


def magnitude_label(effect: float, kind: str) -> str:
    """Conventional (heuristic) magnitude labels for common effect sizes."""
    if effect is None or (isinstance(effect, float) and math.isnan(effect)):
        return "n/a"
    e = abs(effect)
    thresholds = {
        "cohens_h": (0.2, 0.5, 0.8),
        "cohens_d": (0.2, 0.5, 0.8),
        "rank_biserial": (0.1, 0.3, 0.5),
        "cramers_v": (0.1, 0.3, 0.5),
        "spearman_rho": (0.1, 0.3, 0.5),
        "epsilon_squared": (0.01, 0.08, 0.26),
    }[kind]
    if e < thresholds[0]:
        return "negligible"
    if e < thresholds[1]:
        return "small"
    if e < thresholds[2]:
        return "medium"
    return "large"


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def fisher_greater(k1: int, n1: int, k_total: int, n_total: int) -> float:
    """One-sided Fisher exact test that the subgroup event rate exceeds the complement's.

    Equivalent to ``scipy.stats.fisher_exact(..., alternative="greater")`` on the 2×2 table
    [subgroup, complement] × [event, no event], computed via the hypergeometric survival
    function: P(X >= k1) with X ~ Hypergeom(N=n_total, K=k_total, n=n1).
    """
    if n1 <= 0 or n1 >= n_total:
        return 1.0
    return float(min(1.0, max(0.0, sps.hypergeom.sf(k1 - 1, n_total, k_total, n1))))


def welch_greater(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided Welch t-test that mean(a) > mean(b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    if a.var() == 0 and b.var() == 0:
        return 0.0 if a.mean() > b.mean() else 1.0
    res = sps.ttest_ind(a, b, equal_var=False, alternative="greater")
    p = float(res.pvalue)
    return 1.0 if math.isnan(p) else p


# --------------------------------------------------------------------------------------
# Group statistics
# --------------------------------------------------------------------------------------


def binary_group_stats(mask: np.ndarray, events: np.ndarray, *, inference: str = "descriptive",
                       with_tests: bool = True) -> GroupStats:
    """Statistics for a subgroup of a population with a binary event (e.g. error)."""
    mask = np.asarray(mask, dtype=bool)
    events = np.asarray(events, dtype=float)
    N = len(events)
    n = int(mask.sum())
    k_total = int(events.sum())
    k = int(events[mask].sum())
    n_c, k_c = N - n, k_total - k
    rate = k / n if n else float("nan")
    rate_c = k_c / n_c if n_c else float("nan")
    base = k_total / N if N else float("nan")
    stats = GroupStats(
        n=n,
        n_events=k,
        value=rate,
        baseline=base,
        complement_value=rate_c,
        lift=rate / base if base > 0 else float("nan"),
        coverage=n / N if N else float("nan"),
        event_coverage=k / k_total if k_total else float("nan"),
        population_size=N,
        inference=inference,
    )
    if with_tests and n > 0:
        stats.ci_low, stats.ci_high = wilson_interval(k, n)
        if n_c > 0:
            stats.relative_risk, stats.rr_ci_low, stats.rr_ci_high = relative_risk_interval(
                k, n, k_c, n_c)
            stats.effect_size = cohens_h(rate, rate_c)
            stats.effect_size_name = "Cohen's h"
        stats.p_value = fisher_greater(k, n, k_total, N)
        stats.test = "one-sided Fisher exact (hypergeometric)"
    return stats


def continuous_group_stats(mask: np.ndarray, values: np.ndarray, *,
                           inference: str = "descriptive", with_tests: bool = True) -> GroupStats:
    """Statistics for a subgroup of a population with a continuous loss (e.g. |residual|)."""
    mask = np.asarray(mask, dtype=bool)
    values = np.asarray(values, dtype=float)
    N = len(values)
    n = int(mask.sum())
    inside, outside = values[mask], values[~mask]
    mean = float(inside.mean()) if n else float("nan")
    mean_c = float(outside.mean()) if len(outside) else float("nan")
    base = float(values.mean()) if N else float("nan")
    total = float(values.sum())
    stats = GroupStats(
        n=n,
        value=mean,
        baseline=base,
        complement_value=mean_c,
        lift=mean / base if base > 0 else float("nan"),
        coverage=n / N if N else float("nan"),
        event_coverage=float(inside.sum()) / total if total > 0 else float("nan"),
        population_size=N,
        inference=inference,
    )
    if with_tests and n > 1:
        stats.ci_low, stats.ci_high = mean_interval(inside)
        stats.relative_risk = mean / mean_c if mean_c > 0 else float("nan")
        stats.effect_size = cohens_d(inside, outside)
        stats.effect_size_name = "Cohen's d"
        stats.p_value = welch_greater(inside, outside)
        stats.test = "one-sided Welch t-test"
    return stats


# --------------------------------------------------------------------------------------
# Multiple testing
# --------------------------------------------------------------------------------------


def adjust_pvalues(pvalues: np.ndarray, method: str, n_tests: int | None = None) -> np.ndarray:
    """Adjust p-values for multiple testing.

    Args:
        pvalues: Raw p-values of the hypotheses actually computed.
        method: ``"fdr_bh"``, ``"fdr_by"``, ``"holm"``, ``"bonferroni"`` or ``"none"``.
        n_tests: Size of the full family of hypotheses. When larger than ``len(pvalues)``,
            the missing hypotheses are treated as having p = 1. This is how ErrorLens
            accounts for every candidate evaluated during an (in-sample) search even though
            only the most promising ones are tested exactly — a conservative choice.

    Returns:
        Adjusted p-values (same order as the input), capped at 1.
    """
    p = np.asarray(pvalues, dtype=float)
    k = len(p)
    if k == 0:
        return p.copy()
    m = max(int(n_tests or k), k)
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(k)) * ranked)
    elif method in ("fdr_bh", "fdr_by"):
        factor = 1.0
        if method == "fdr_by":
            factor = float(np.sum(1.0 / np.arange(1, m + 1)))
        ranks = np.arange(1, k + 1)
        raw = ranked * m * factor / ranks
        adj = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ValueError(f"Unknown correction method {method!r}")
    out = np.empty(k)
    out[order] = np.minimum(adj, 1.0)
    return out
