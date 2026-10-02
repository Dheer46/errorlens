"""Candidate generators: vectorized beam search and a surrogate error tree.

Both generators only *propose* candidate subgroups. Whether a candidate is reported is
decided later by validation tests, multiple-testing correction, ranking and de-duplication
(see :mod:`errorlens.discovery.subgroup`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from errorlens.core.types import Condition, FeatureInfo
from errorlens.discovery.items import (
    ItemSet,
    native,
    nice_threshold,
    numeric_values,
    top_levels,
)


@dataclass
class SearchOutput:
    candidates: list[tuple[Condition, ...]]
    n_evaluated: int


def quality(support: np.ndarray, mean: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Standardized excess ``sqrt(n) * (mean - mu) / sd`` (binomial / z quality function)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(support) * (mean - mu) / sd


def beam_search(items: ItemSet, target: np.ndarray, *, min_support: int, max_depth: int,
                beam_width: int, pool_size: int) -> SearchOutput:
    """Level-wise beam search over conjunctions of items on distinct features.

    Each level extends the ``beam_width`` best subgroups of the previous level by one item.
    Support and target sums for *all* extensions of all beam members are obtained with a
    single matrix product, so the cost is linear in the number of rows.
    """
    n, m = items.matrix.shape
    t = np.asarray(target, dtype=np.float64)
    if m == 0 or n == 0:
        return SearchOutput([], 0)
    mu = float(t.mean())
    sd = float(t.std())
    if sd <= 0:
        return SearchOutput([], 0)

    M = items.matrix.astype(np.float32)
    feat = items.feature_index
    pool: dict[frozenset[int], float] = {}
    n_evaluated = 0

    # Level 1
    support = M.sum(axis=0).astype(np.float64)
    sums = (t.astype(np.float32) @ M).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / support
    q = quality(support, means, mu, sd)
    valid = (support >= min_support) & (means > mu)
    n_evaluated += int((support >= min_support).sum())
    for j in np.flatnonzero(valid):
        pool[frozenset([int(j)])] = float(q[j])
    order = np.flatnonzero(valid)[np.argsort(-q[valid])][:beam_width]
    beam: list[tuple[tuple[int, ...], np.ndarray]] = [
        ((int(j),), items.matrix[:, j].copy()) for j in order
    ]

    for _depth in range(2, max_depth + 1):
        if not beam:
            break
        # Stack [mask, mask * t] for all beam members and compute all extensions at once.
        W = np.empty((2 * len(beam), n), dtype=np.float32)
        for b, (_, mask) in enumerate(beam):
            W[2 * b] = mask
            W[2 * b + 1] = mask * t
        S = W @ M  # (2B, m)
        level: dict[frozenset[int], tuple[float, tuple[int, ...], int]] = {}
        for b, (members, _) in enumerate(beam):
            sup = S[2 * b].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = S[2 * b + 1].astype(np.float64) / sup
            used = np.isin(feat, feat[list(members)])
            ok_support = (sup >= min_support) & ~used
            n_evaluated += int(ok_support.sum())
            qq = quality(sup, mean, mu, sd)
            ok = ok_support & (mean > mu)
            for j in np.flatnonzero(ok):
                key = frozenset((*members, int(j)))
                if key in level or key in pool:
                    continue
                level[key] = (float(qq[j]), (*members, int(j)), b)
        if not level:
            break
        for key, (qv, _, _) in level.items():
            pool[key] = qv
        best = sorted(level.values(), key=lambda v: -v[0])[:beam_width]
        beam = [(members, beam[b][1] & items.matrix[:, members[-1]])
                for _, members, b in best]

    top = sorted(pool.items(), key=lambda kv: -kv[1])[:pool_size]
    candidates = [tuple(items.conditions[j] for j in sorted(key)) for key, _ in top]
    return SearchOutput(candidates, n_evaluated)


# --------------------------------------------------------------------------------------
# Surrogate error tree
# --------------------------------------------------------------------------------------


@dataclass
class _EncodedColumn:
    feature: str
    kind: str  # "numeric", "level", "missing"
    level: object = None
    integer_valued: bool = False


def _encode(X: pd.DataFrame, infos: list[FeatureInfo], max_categories: int,
            min_support: int) -> tuple[np.ndarray, list[_EncodedColumn]]:
    cols: list[np.ndarray] = []
    meta: list[_EncodedColumn] = []
    for info in infos:
        if info.kind == "skipped":
            continue
        col = X[info.name]
        if info.kind == "numeric":
            values = numeric_values(col)
            nan = np.isnan(values)
            present = values[~nan]
            fill = float(np.median(present)) if len(present) else 0.0
            cols.append(np.where(nan, fill, values))
            meta.append(_EncodedColumn(info.name, "numeric",
                                       integer_valued=bool(np.all(np.mod(present, 1) == 0))))
        else:
            for level in top_levels(col, max_categories, min_support):
                cols.append((col == level).to_numpy(dtype=float, na_value=0.0))
                meta.append(_EncodedColumn(info.name, "level", level=native(level)))
        if info.n_missing >= min_support:
            cols.append(col.isna().to_numpy(dtype=float))
            meta.append(_EncodedColumn(info.name, "missing"))
    if not cols:
        return np.zeros((len(X), 0)), meta
    return np.column_stack(cols), meta


def tree_search(X: pd.DataFrame, infos: list[FeatureInfo], target: np.ndarray, *,
                is_binary: bool, min_support: int, max_depth: int, max_categories: int,
                random_state: int | None) -> SearchOutput:
    """Fit a shallow tree to the error target and return above-baseline node paths."""
    Z, meta = _encode(X, infos, max_categories, min_support)
    t = np.asarray(target, dtype=float)
    if Z.shape[1] == 0 or len(np.unique(t)) < 2:
        return SearchOutput([], 0)
    model: DecisionTreeClassifier | DecisionTreeRegressor
    if is_binary:
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_support,
                                       random_state=random_state)
        model.fit(Z, t.astype(int))
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_support,
                                      random_state=random_state)
        model.fit(Z, t)
    tree = model.tree_
    mu = float(t.mean())
    # node value: mean target in node
    if is_binary:
        vals = tree.value[:, 0, :]
        totals = vals.sum(axis=1)
        node_mean = vals[:, 1] / np.where(totals > 0, totals, 1) if vals.shape[1] > 1 else (
            np.zeros(len(vals)))
    else:
        node_mean = tree.value[:, 0, 0]

    candidates: list[tuple[Condition, ...]] = []
    stack: list[tuple[int, list[tuple[int, bool, float]]]] = [(0, [])]
    while stack:
        node, path = stack.pop()
        if node != 0 and node_mean[node] > mu and tree.n_node_samples[node] >= min_support:
            conds = _path_to_conditions(path, meta)
            if conds:
                candidates.append(conds)
        left, right = tree.children_left[node], tree.children_right[node]
        if left != -1:
            f, th = int(tree.feature[node]), float(tree.threshold[node])
            stack.append((left, [*path, (f, True, th)]))
            stack.append((right, [*path, (f, False, th)]))
    return SearchOutput(candidates, len(candidates))


def _path_to_conditions(path: list[tuple[int, bool, float]],
                        meta: list[_EncodedColumn]) -> tuple[Condition, ...]:
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    integer: dict[str, bool] = {}
    eq: dict[str, object] = {}
    neq: dict[str, list[object]] = {}
    missing: dict[str, bool] = {}
    for f, is_left, th in path:
        col = meta[f]
        if col.kind == "numeric":
            integer[col.feature] = col.integer_valued
            if is_left:
                upper[col.feature] = min(upper.get(col.feature, math.inf), th)
            else:
                lower[col.feature] = max(lower.get(col.feature, -math.inf), th)
        elif col.kind == "level":
            if is_left:
                neq.setdefault(col.feature, []).append(col.level)
            else:
                eq[col.feature] = col.level
        else:
            missing[col.feature] = not is_left
    conds: list[Condition] = []
    features = list(dict.fromkeys([meta[f].feature for f, _, _ in path]))
    for feat in features:
        if missing.get(feat) is True:
            conds.append(Condition(feat, "is_missing"))
            continue
        if feat in lower or feat in upper:
            iv = integer.get(feat, False)
            lo = nice_threshold(lower[feat], iv) if feat in lower else None
            hi = nice_threshold(upper[feat], iv) if feat in upper else None
            if lo is not None and hi is not None:
                if lo < hi:
                    conds.append(Condition(feat, "in_range", lo, hi))
            elif hi is not None:
                conds.append(Condition(feat, "<=", hi))
            elif lo is not None:
                conds.append(Condition(feat, ">", lo))
        if feat in eq:
            conds.append(Condition(feat, "==", eq[feat]))
        elif feat in neq:
            conds.extend(Condition(feat, "!=", lv) for lv in neq[feat])
        if missing.get(feat) is False and feat not in lower and feat not in upper \
                and feat not in eq and feat not in neq:
            conds.append(Condition(feat, "not_missing"))
    return tuple(conds)


# --------------------------------------------------------------------------------------
# Threshold refinement
# --------------------------------------------------------------------------------------

REFINE_BINS = 40


def refine_thresholds(conditions: tuple[Condition, ...], X: pd.DataFrame, target: np.ndarray,
                      *, min_support: int, mu: float, sd: float) -> tuple[Condition, ...]:
    """Locally optimize each numeric threshold on a finer quantile grid.

    The beam search only sees ``n_bins`` thresholds per feature. For every ``<=`` / ``>``
    condition, the threshold is re-chosen among ``REFINE_BINS`` readable quantiles of the
    feature *within the rows selected by the other conditions*, maximizing the same quality
    function. Runs on the discovery split only, so validation remains independent.
    """
    if sd <= 0:
        return conditions
    conds = list(conditions)
    t = np.asarray(target, dtype=float)
    for i, cond in enumerate(conds):
        if cond.op not in ("<=", ">"):
            continue
        others = np.ones(len(X), dtype=bool)
        for j, c in enumerate(conds):
            if j != i:
                others &= c.mask(X)
        values = numeric_values(X[cond.feature])
        sel = others & ~np.isnan(values)
        if sel.sum() < 2 * min_support:
            continue
        v, tv = values[sel], t[sel]
        integer_valued = bool(np.all(np.mod(v, 1) == 0))
        grid = {nice_threshold(float(q), integer_valued)
                for q in np.quantile(v, np.linspace(0, 1, REFINE_BINS + 1)[1:-1])}
        grid.add(float(cond.value))
        thresholds = np.array(sorted(grid))
        order = np.argsort(v, kind="stable")
        v_sorted, csum = v[order], np.cumsum(tv[order])
        n_le = np.searchsorted(v_sorted, thresholds, side="right")
        sum_le = np.where(n_le > 0, csum[np.maximum(n_le - 1, 0)], 0.0)
        if cond.op == "<=":
            n_arr, s_arr = n_le, sum_le
        else:
            n_arr, s_arr = len(v) - n_le, csum[-1] - sum_le
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.sqrt(n_arr) * (s_arr / n_arr - mu) / sd
        q[n_arr < min_support] = -np.inf
        if not np.isfinite(q).any():
            continue
        best_t = float(thresholds[int(np.argmax(q))])
        conds[i] = Condition(cond.feature, cond.op, best_t)
    return tuple(conds)
