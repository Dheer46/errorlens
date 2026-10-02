"""Errors-vs-correct comparison: which features are *associated* with model errors.

Classification compares the feature distribution of misclassified rows with correctly
classified rows. Regression relates each feature to the absolute error. All p-values are
Benjamini–Hochberg adjusted across features. Effect sizes and their magnitude labels are
reported because, with large samples, tiny differences become "significant".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from errorlens.analysis.statistics import adjust_pvalues, magnitude_label
from errorlens.core.types import FeatureAssociation, FeatureInfo
from errorlens.utils.formatting import fmt_num

MAX_LEVELS = 20
N_BINS = 5
MISSING = "(missing)"
OTHER = "(other)"


def feature_associations(X: pd.DataFrame, infos: list[FeatureInfo], *, errors: np.ndarray | None,
                         abs_error: np.ndarray | None) -> list[FeatureAssociation]:
    """Compute one :class:`FeatureAssociation` per usable feature.

    Pass ``errors`` (boolean) for classification or ``abs_error`` for regression.
    """
    out: list[FeatureAssociation] = []
    for info in infos:
        if info.kind == "skipped":
            continue
        col = X[info.name]
        if errors is not None:
            assoc = (_numeric_classification(info.name, col, errors) if info.kind == "numeric"
                     else _categorical_classification(info.name, info.kind, col, errors))
        else:
            assert abs_error is not None
            assoc = (_numeric_regression(info.name, col, abs_error) if info.kind == "numeric"
                     else _categorical_regression(info.name, info.kind, col, abs_error))
        if assoc is not None:
            out.append(assoc)
    if out:
        adj = adjust_pvalues(np.array([a.p_value if not math.isnan(a.p_value) else 1.0
                                       for a in out]), "fdr_bh")
        for a, p in zip(out, adj):
            a.p_value_adjusted = float(p)
    out.sort(key=_sort_value, reverse=True)
    return out


def _sort_value(a: FeatureAssociation) -> float:
    if math.isnan(a.effect_size):
        return -1.0
    if a.effect_size_name == "epsilon_squared":
        return math.sqrt(max(a.effect_size, 0.0))
    return abs(a.effect_size)


# --------------------------------------------------------------------------------------
# Level / bin tables (used for "error rate by feature" plots and report tables)
# --------------------------------------------------------------------------------------


def _level_table(keys: pd.Series, metric: np.ndarray, order: list[Any] | None = None
                 ) -> list[dict[str, Any]]:
    df = pd.DataFrame({"key": keys.astype(object).to_numpy(), "m": metric})
    grouped = df.groupby("key", sort=False, dropna=False)["m"].agg(["count", "mean"])
    if order is not None:
        grouped = grouped.reindex([k for k in order if k in grouped.index])
    return [{"level": str(k), "n": int(row["count"]), "value": float(row["mean"])}
            for k, row in grouped.iterrows()]


def _numeric_bins(x: np.ndarray) -> tuple[pd.Series, list[str]]:
    present = x[~np.isnan(x)]
    n_unique = len(np.unique(present))
    labels: pd.Series
    if n_unique <= N_BINS:
        labels = pd.Series(np.where(np.isnan(x), MISSING, pd.Series(x).map(fmt_num)))
        order = [fmt_num(v) for v in np.unique(present)] + [MISSING]
        return labels, order
    edges = np.unique(np.quantile(present, np.linspace(0, 1, N_BINS + 1)))
    codes = np.searchsorted(edges[1:-1], x, side="left")
    names = []
    for i in range(len(edges) - 1):
        lo = "min" if i == 0 else fmt_num(edges[i])
        names.append(f"{lo} – {fmt_num(edges[i + 1])}")
    lab = np.array([names[min(c, len(names) - 1)] for c in codes], dtype=object)
    lab[np.isnan(x)] = MISSING
    return pd.Series(lab), [*names, MISSING]


def _categorical_keys(col: pd.Series) -> tuple[pd.Series, list[Any]]:
    counts = col.value_counts(dropna=True)
    top = list(counts.index[:MAX_LEVELS])
    keys = col.astype(object).where(col.isin(top), OTHER)
    keys = keys.where(col.notna(), MISSING)
    return keys.reset_index(drop=True), [*top, OTHER, MISSING]


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def _numeric_classification(name: str, col: pd.Series, errors: np.ndarray
                            ) -> FeatureAssociation | None:
    x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ok = ~np.isnan(x)
    a, b = x[ok & errors], x[ok & ~errors]
    if len(a) < 2 or len(b) < 2:
        return None
    res = sps.mannwhitneyu(a, b, alternative="two-sided")
    r = 2.0 * float(res.statistic) / (len(a) * len(b)) - 1.0
    keys, order = _numeric_bins(x)
    direction = "higher values among errors" if r > 0 else "lower values among errors"
    return FeatureAssociation(
        feature=name, feature_type="numeric", test="Mann-Whitney U",
        statistic=float(res.statistic), p_value=float(res.pvalue), effect_size=r,
        effect_size_name="rank_biserial", magnitude=magnitude_label(r, "rank_biserial"),
        summary_errors=f"mean {fmt_num(float(a.mean()))}, median {fmt_num(float(np.median(a)))}",
        summary_correct=f"mean {fmt_num(float(b.mean()))}, median {fmt_num(float(np.median(b)))}",
        direction=direction if abs(r) >= 0.1 else "similar distributions",
        levels=_level_table(keys, errors.astype(float), order),
    )


def _categorical_classification(name: str, kind: str, col: pd.Series, errors: np.ndarray
                                ) -> FeatureAssociation | None:
    keys, order = _categorical_keys(col)
    table = pd.crosstab(keys, errors)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    # Pool sparse levels so the chi-square approximation is reasonable.
    expected_min = table.sum(axis=1) * table.sum(axis=0).min() / table.to_numpy().sum()
    # Positional masks: level labels may themselves be booleans, which .loc would misread.
    sparse = (expected_min < 5).to_numpy()
    if sparse.any() and not sparse.all():
        pooled = table.iloc[sparse].sum()
        table = table.iloc[~sparse]
        table = pd.concat([table, pooled.to_frame("(pooled sparse)").T])
    if table.shape[0] < 2:
        return None
    chi2, p, _, _ = sps.chi2_contingency(table.to_numpy(), correction=False)
    n = table.to_numpy().sum()
    v = math.sqrt(chi2 / n) if n else float("nan")
    levels = _level_table(keys, errors.astype(float), order)
    worst = max(levels, key=lambda d: d["value"] if d["n"] >= 10 else -1)
    return FeatureAssociation(
        feature=name, feature_type=kind, test="Chi-square test of independence",
        statistic=float(chi2), p_value=float(p), effect_size=v, effect_size_name="cramers_v",
        magnitude=magnitude_label(v, "cramers_v"),
        summary_errors=f"highest error rate: {worst['level']} ({worst['value']:.1%})",
        summary_correct=f"{len(levels)} levels",
        direction="error rate differs across levels" if v >= 0.1 else "similar error rates",
        levels=levels,
    )


# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------


def _numeric_regression(name: str, col: pd.Series, abs_error: np.ndarray
                        ) -> FeatureAssociation | None:
    x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ok = ~np.isnan(x)
    if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(abs_error[ok]) == 0:
        return None
    res = sps.spearmanr(x[ok], abs_error[ok])
    rho = float(res.statistic)
    keys, order = _numeric_bins(x)
    return FeatureAssociation(
        feature=name, feature_type="numeric", test="Spearman correlation with |error|",
        statistic=rho, p_value=float(res.pvalue), effect_size=rho,
        effect_size_name="spearman_rho", magnitude=magnitude_label(rho, "spearman_rho"),
        direction=("larger errors at higher values" if rho > 0 else
                   "larger errors at lower values") if abs(rho) >= 0.1 else "no clear trend",
        levels=_level_table(keys, abs_error, order),
    )


def _categorical_regression(name: str, kind: str, col: pd.Series, abs_error: np.ndarray
                            ) -> FeatureAssociation | None:
    keys, order = _categorical_keys(col)
    groups = [abs_error[(keys == k).to_numpy()] for k in pd.unique(keys)]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 2:
        return None
    if np.ptp(np.concatenate(groups)) == 0:
        return None
    h, p = sps.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = float(h) / (n - 1) if n > 1 else float("nan")
    levels = _level_table(keys, abs_error, order)
    worst = max(levels, key=lambda d: d["value"] if d["n"] >= 10 else -1)
    return FeatureAssociation(
        feature=name, feature_type=kind, test="Kruskal-Wallis on |error|",
        statistic=float(h), p_value=float(p), effect_size=eps2,
        effect_size_name="epsilon_squared", magnitude=magnitude_label(eps2, "epsilon_squared"),
        summary_errors=f"highest MAE: {worst['level']} ({fmt_num(worst['value'])})",
        direction="error magnitude differs across levels" if eps2 >= 0.01 else "similar errors",
        levels=levels,
    )
