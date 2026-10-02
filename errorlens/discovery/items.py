"""Turn raw features into a vocabulary of interpretable candidate conditions ("items")."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from errorlens.core.types import Condition, FeatureInfo


@dataclass
class ItemSet:
    """Candidate conditions plus their boolean membership matrix on the discovery data."""

    conditions: list[Condition]
    matrix: np.ndarray  # (n_rows, n_items) bool
    feature_index: np.ndarray  # (n_items,) int — which feature each item constrains
    features: list[str]

    @property
    def n_items(self) -> int:
        return len(self.conditions)


def nice_threshold(value: float, integer_valued: bool) -> float:
    """Round a threshold to 3 significant digits (or down to an integer for integer data)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    digits = 3 - int(math.floor(math.log10(abs(value)))) - 1
    rounded = float(round(value, digits))
    return float(math.floor(rounded)) if integer_valued else rounded


def numeric_values(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def numeric_thresholds(values: np.ndarray, n_bins: int) -> list[float]:
    """Readable quantile thresholds for a numeric column (NaNs ignored)."""
    present = values[~np.isnan(values)]
    if len(present) == 0:
        return []
    uniques = np.unique(present)
    if len(uniques) < 2:
        return []
    integer_valued = bool(np.all(np.mod(present, 1) == 0))
    if len(uniques) <= n_bins:
        raw = list(uniques[:-1]) if integer_valued else list((uniques[:-1] + uniques[1:]) / 2)
    else:
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        raw = list(np.quantile(present, qs))
    out = sorted({nice_threshold(float(v), integer_valued) for v in raw})
    lo, hi = float(uniques[0]), float(uniques[-1])
    return [t for t in out if lo <= t < hi]


def top_levels(col: pd.Series, max_categories: int, min_support: int) -> list[object]:
    counts = col.value_counts(dropna=True)
    counts = counts[counts >= min_support]
    return list(counts.index[:max_categories])


def build_items(X: pd.DataFrame, infos: list[FeatureInfo], *, n_bins: int,
                max_categories: int, min_support: int) -> ItemSet:
    """Build candidate conditions for all usable features.

    * numeric: ``x <= t`` and ``x > t`` at readable quantile thresholds;
    * boolean / categorical: ``x == level`` for frequent levels;
    * any feature with enough missing values: ``x is missing``.

    Items whose support is below ``min_support`` or that select every row are dropped.
    """
    n = len(X)
    conditions: list[Condition] = []
    columns: list[np.ndarray] = []
    feat_idx: list[int] = []
    features: list[str] = []

    def add(cond: Condition, mask: np.ndarray, fi: int) -> None:
        support = int(mask.sum())
        if min_support <= support < n:
            conditions.append(cond)
            columns.append(mask)
            feat_idx.append(fi)

    for info in infos:
        if info.kind == "skipped":
            continue
        fi = len(features)
        features.append(info.name)
        col = X[info.name]
        if info.kind == "numeric":
            values = numeric_values(col)
            notnan = ~np.isnan(values)
            for t in numeric_thresholds(values, n_bins):
                with np.errstate(invalid="ignore"):
                    add(Condition(info.name, "<=", t), notnan & (values <= t), fi)
                    add(Condition(info.name, ">", t), notnan & (values > t), fi)
        else:
            for level in top_levels(col, max_categories, min_support):
                cond = Condition(info.name, "==", native(level))
                add(cond, cond.mask(X), fi)
        if info.n_missing >= min_support:
            cond = Condition(info.name, "is_missing")
            add(cond, cond.mask(X), fi)

    matrix = np.column_stack(columns) if columns else np.zeros((n, 0), dtype=bool)
    return ItemSet(conditions, matrix.astype(bool, copy=False),
                   np.asarray(feat_idx, dtype=int), features)


def native(value: object) -> object:
    """Convert numpy scalars to Python scalars so conditions serialize cleanly."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value
