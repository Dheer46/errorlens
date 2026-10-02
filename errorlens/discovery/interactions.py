"""Two-feature error grids (heatmaps) and feature pairs that co-occur in failure patterns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from errorlens.core.types import FailurePattern, FeatureInfo, to_jsonable
from errorlens.utils.formatting import fmt_num

MISSING = "(missing)"


@dataclass
class ErrorGrid:
    """Error metric for every cell of a two-feature grid."""

    feature_x: str
    feature_y: str
    x_labels: list[str]
    y_labels: list[str]
    values: np.ndarray  # (len(y_labels), len(x_labels)); NaN where too few rows
    counts: np.ndarray
    metric_name: str
    baseline: float
    min_count: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.y_labels, columns=self.x_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_x": self.feature_x,
            "feature_y": self.feature_y,
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "values": to_jsonable(self.values),
            "counts": to_jsonable(self.counts),
            "metric_name": self.metric_name,
            "baseline": to_jsonable(self.baseline),
        }


def _discretize(col: pd.Series, kind: str, bins: int) -> tuple[np.ndarray, list[str]]:
    if kind == "numeric":
        x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        present = x[~np.isnan(x)]
        uniques = np.unique(present)
        if len(uniques) <= bins:
            labels = [fmt_num(v) for v in uniques]
            codes = np.searchsorted(uniques, x)
        else:
            edges = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)))
            codes = np.searchsorted(edges[1:-1], x, side="left")
            labels = [f"≤ {fmt_num(edges[i + 1])}" for i in range(len(edges) - 1)]
        out = np.where(np.isnan(x), -1, np.minimum(codes, len(labels) - 1))
        if np.isnan(x).any():
            labels = [*labels, MISSING]
            out = np.where(out == -1, len(labels) - 1, out)
        return out.astype(int), labels
    counts = col.value_counts(dropna=True)
    top = list(counts.index[:bins])
    labels = [str(v) for v in top]
    mapping = {v: i for i, v in enumerate(top)}
    codes = col.map(mapping)
    if (codes.isna() & col.notna()).any():
        labels.append("(other)")
        codes = codes.where(~(codes.isna() & col.notna()), len(labels) - 1)
    if col.isna().any():
        labels.append(MISSING)
        codes = codes.where(col.notna(), len(labels) - 1)
    return codes.to_numpy(dtype=int), labels


def error_grid(X: pd.DataFrame, metric: np.ndarray, feature_x: str, feature_y: str,
               infos: list[FeatureInfo], *, bins: int = 6, metric_name: str = "error rate",
               min_count: int = 10) -> ErrorGrid:
    """Mean of ``metric`` (error indicator or absolute error) in each cell of a 2-D grid."""
    kinds = {i.name: i.kind for i in infos}
    cx, lx = _discretize(X[feature_x], kinds.get(feature_x, "categorical"), bins)
    cy, ly = _discretize(X[feature_y], kinds.get(feature_y, "categorical"), bins)
    sums = np.zeros((len(ly), len(lx)))
    counts = np.zeros((len(ly), len(lx)), dtype=int)
    np.add.at(sums, (cy, cx), metric)
    np.add.at(counts, (cy, cx), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(counts >= min_count, sums / np.maximum(counts, 1), np.nan)
    return ErrorGrid(feature_x, feature_y, lx, ly, values, counts, metric_name,
                     float(np.mean(metric)), min_count)


def top_feature_pair(patterns: list[FailurePattern], fallback: list[str]) -> tuple[str, str] | None:
    """The pair of features that most often appear together in the top patterns.

    Falls back to the two first features of ``fallback`` (e.g. features most associated with
    errors) when no multi-feature pattern exists.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for p in patterns[:10]:
        feats = sorted({c.feature for c in p.conditions})
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pairs[(feats[i], feats[j])] += 1
    if pairs:
        return pairs.most_common(1)[0][0]
    singles = list(dict.fromkeys([c.feature for p in patterns[:10] for c in p.conditions]
                                 + fallback))
    if len(singles) >= 2:
        return (singles[0], singles[1])
    return None
