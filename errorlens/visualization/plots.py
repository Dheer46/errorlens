"""Matplotlib plots for an :class:`~errorlens.core.results.AnalysisResult`.

Every public function returns a matplotlib ``Figure``. Each has an ``_draw_*`` counterpart
operating on an existing ``Axes`` so the same drawing code serves interactive use, the
overview dashboard and report figures.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from errorlens.core.types import ClassificationPerformance, FailurePattern, FeatureAssociation
from errorlens.discovery.interactions import error_grid
from errorlens.exceptions import InvalidInputError, UnsupportedTaskError
from errorlens.utils.formatting import fmt_num, fmt_pct
from errorlens.visualization._mpl import (
    BASELINE,
    BLUE,
    MUTED,
    ORANGE,
    STYLE,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    mpl,
    new_figure,
    sequential_cmap,
)

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult


def _style() -> Any:
    return mpl().rc_context(STYLE)


def _fmt_metric(value: float, is_rate: bool) -> str:
    return fmt_pct(value) if is_rate else fmt_num(value)


def _empty(ax: Any, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=9,
            transform=ax.transAxes, wrap=True)


# --------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------


def _patterns_for(result: AnalysisResult, kind: str | None) -> list[FailurePattern]:
    if kind is None or kind == result.primary_kind:
        return result.patterns
    if kind in result.directional_patterns:
        return result.directional_patterns[kind]
    raise InvalidInputError(
        f"Unknown pattern kind {kind!r}; available: "
        f"{[result.primary_kind, *result.directional_patterns]}")


def _draw_patterns(ax: Any, patterns: list[FailurePattern], top: int, title: str) -> None:
    if not patterns:
        _empty(ax, "No statistically significant failure pattern")
        ax.set_title(title)
        return
    pats = patterns[:top][::-1]
    is_rate = pats[0].is_binary
    y = np.arange(len(pats))
    values = np.array([p.rate for p in pats])
    lo = np.array([p.validation.ci_low for p in pats])
    hi = np.array([p.validation.ci_high for p in pats])
    ax.barh(y, values, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    err = np.vstack([np.clip(values - lo, 0, None), np.clip(hi - values, 0, None)])
    ax.errorbar(values, y, xerr=err, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1,
                capsize=2.5, zorder=3)
    base = pats[0].baseline
    ax.axvline(base, color=BASELINE, linestyle="--", linewidth=1, zorder=4)
    ax.text(base, 0.995, f" baseline {_fmt_metric(base, is_rate)}", color=TEXT_SECONDARY,
            fontsize=7.5, va="top", transform=ax.get_xaxis_transform())
    labels = [f"#{p.rank}  " + "\n".join(textwrap.wrap(p.description, 38)) for p in pats]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    for yi, p in zip(y, pats):
        ax.text(max(p.validation.ci_high, p.rate), yi, f"  {p.lift:.2f}×",
                va="center", fontsize=7.5, color=TEXT)
    xmax = max(float(np.nanmax(hi)), float(np.nanmax(values))) * 1.18
    ax.set_xlim(0, xmax if np.isfinite(xmax) and xmax > 0 else 1)
    if is_rate:
        ax.xaxis.set_major_formatter(mpl().ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f"{pats[0].metric_name} (bar: full data; whisker: 95% CI, "
                  f"{'held-out' if pats[0].validation.inference == 'holdout' else 'in-sample'})")
    ax.set_title(title)


def plot_patterns(result: AnalysisResult, kind: str | None = None, top: int = 10,
                  ax: Any = None) -> Any:
    """Horizontal bars of each pattern's error metric with confidence intervals."""
    patterns = _patterns_for(result, kind)
    with _style():
        if ax is None:
            n = max(min(len(patterns), top), 3)
            fig, ax = new_figure((8.0, 1.0 + 0.55 * n), managed=True)
        else:
            fig = ax.figure
        title = {"error": "Top failure patterns", "high_error": "Highest-error regions",
                 "false_positive": "False-positive patterns",
                 "false_negative": "False-negative patterns",
                 "underprediction": "Severe underprediction patterns",
                 "overprediction": "Severe overprediction patterns"}.get(
            kind or result.primary_kind, "Failure patterns")
        _draw_patterns(ax, patterns, top, title)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Feature-level error rates
# --------------------------------------------------------------------------------------


def _association(result: AnalysisResult, feature: str | None,
                 numeric_only: bool = False) -> FeatureAssociation | None:
    pool = [a for a in result.feature_analysis
            if not numeric_only or a.feature_type == "numeric"]
    if feature is None:
        return pool[0] if pool else None
    for a in result.feature_analysis:
        if a.feature == feature:
            return a
    if feature not in result.X.columns:
        raise InvalidInputError(f"Feature {feature!r} not found.")
    return None


def _draw_feature_errors(ax: Any, result: AnalysisResult, assoc: FeatureAssociation | None
                         ) -> None:
    if assoc is None or not assoc.levels:
        _empty(ax, "No feature analysis available")
        return
    levels = [lv for lv in assoc.levels if lv["n"] > 0]
    x = np.arange(len(levels))
    values = np.array([lv["value"] for lv in levels])
    is_rate = result.is_classification
    ax.bar(x, values, width=0.7, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2)
    base = (result.performance.error_rate if isinstance(result.performance,
                                                        ClassificationPerformance)
            else float(np.abs(result.y_true - result.y_pred).mean()))
    ax.axhline(base, color=BASELINE, linestyle="--", linewidth=1, zorder=3)
    ax.text(len(levels) - 0.5, base, f"overall {_fmt_metric(base, is_rate)}", ha="right",
            va="bottom", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(["\n".join(textwrap.wrap(str(lv["level"]), 12)) for lv in levels],
                       fontsize=7, rotation=0 if len(levels) <= 6 else 45,
                       ha="center" if len(levels) <= 6 else "right")
    ax.grid(axis="x", visible=False)
    if is_rate:
        ax.yaxis.set_major_formatter(mpl().ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("error rate" if is_rate else "MAE")
    ax.set_title(f"{'Error rate' if is_rate else 'MAE'} by {assoc.feature}")
    for xi, lv in zip(x, levels):
        ax.text(xi, lv["value"], f"n={lv['n']:,}", ha="center", va="bottom", fontsize=6.5,
                color=MUTED)


def plot_feature_errors(result: AnalysisResult, feature: str | None = None,
                        ax: Any = None) -> Any:
    """Error rate (classification) or MAE (regression) across levels / bins of features.

    With ``feature=None`` the four features most associated with errors are shown.
    """
    with _style():
        if ax is not None:
            _draw_feature_errors(ax, result, _association(result, feature))
            return ax.figure
        if feature is not None:
            fig, ax = new_figure((6.5, 3.6), managed=True)
            _draw_feature_errors(ax, result, _association(result, feature))
        else:
            assocs = result.feature_analysis[:4]
            n = max(len(assocs), 1)
            cols = 2 if n > 1 else 1
            rows = int(np.ceil(n / cols))
            fig, axes = new_figure((6.0 * cols, 3.4 * rows), managed=True, nrows=rows,
                                   ncols=cols, squeeze=False)
            flat = list(axes.ravel())
            shown: list[FeatureAssociation | None] = list(assocs) or [None]
            for a, axx in zip(shown, flat):
                _draw_feature_errors(axx, result, a)
            for axx in flat[len(shown):]:
                axx.set_axis_off()
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Distributions: errors vs correct / residual histogram
# --------------------------------------------------------------------------------------


def _draw_error_distribution(ax: Any, result: AnalysisResult, feature: str | None) -> None:
    if not result.is_classification:
        r = result.y_true - result.y_pred
        ax.hist(r, bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
        ax.axvline(0, color=BASELINE, linestyle="--", linewidth=1)
        ax.set_xlabel("residual (actual − predicted)")
        ax.set_ylabel("rows")
        ax.set_title("Residual distribution")
        return
    assoc = _association(result, feature, numeric_only=feature is None)
    if assoc is None:
        _empty(ax, "No feature to compare")
        return
    col = result.X[assoc.feature]
    errors = result.y_true != result.y_pred
    if assoc.feature_type == "numeric":
        x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        ok = ~np.isnan(x)
        bins = np.histogram_bin_edges(x[ok], bins=30)
        ax.hist(x[ok & ~errors], bins=bins, density=True, alpha=0.55, color=BLUE,
                label=f"correct (n={int((ok & ~errors).sum()):,})", zorder=2)
        ax.hist(x[ok & errors], bins=bins, density=True, histtype="step", linewidth=2,
                color=ORANGE, label=f"errors (n={int((ok & errors).sum()):,})", zorder=3)
        ax.set_ylabel("density")
        ax.set_xlabel(assoc.feature)
    else:
        keys = col.astype(object).where(col.notna(), "(missing)").astype(str)
        top = keys.value_counts().index[:10]
        share_c = keys[~errors].value_counts(normalize=True).reindex(top, fill_value=0)
        share_e = keys[errors].value_counts(normalize=True).reindex(top, fill_value=0)
        xi = np.arange(len(top))
        ax.bar(xi - 0.2, share_c.to_numpy(), width=0.38, color=BLUE, label="correct", zorder=2)
        ax.bar(xi + 0.2, share_e.to_numpy(), width=0.38, color=ORANGE, label="errors", zorder=2)
        ax.set_xticks(xi)
        ax.set_xticklabels([textwrap.shorten(str(t), 14) for t in top], fontsize=7,
                           rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mpl().ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("share of rows")
        ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title(f"{assoc.feature}: errors vs correct predictions")


def plot_error_distribution(result: AnalysisResult, feature: str | None = None,
                            ax: Any = None) -> Any:
    """Classification: feature distribution of errors vs correct predictions.
    Regression: histogram of residuals."""
    with _style():
        if ax is None:
            fig, ax = new_figure((6.5, 3.6), managed=True)
        else:
            fig = ax.figure
        _draw_error_distribution(ax, result, feature)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------------------


def _draw_confusion(ax: Any, result: AnalysisResult, normalize: bool) -> None:
    perf = result.performance
    assert isinstance(perf, ClassificationPerformance)
    cm = np.asarray(perf.confusion_matrix, dtype=float)
    labels = [str(v) for v in perf.labels]
    if len(labels) > 25:
        _empty(ax, f"{len(labels)} classes — confusion matrix too large to plot")
        return
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    ax.imshow(shown, cmap=sequential_cmap(), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([textwrap.shorten(v, 12) for v in labels], rotation=30 if len(labels) > 4
                       else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_yticklabels([textwrap.shorten(v, 12) for v in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    if len(labels) <= 12:
        vmax = shown.max() if shown.size else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = shown[i, j]
                txt = f"{v:.0%}" if normalize else f"{int(v):,}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * vmax else TEXT)
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))


def plot_confusion_matrix(result: AnalysisResult, normalize: bool = False,
                          ax: Any = None) -> Any:
    if not result.is_classification:
        raise UnsupportedTaskError("plot_confusion_matrix() is only available for classification.")
    with _style():
        if ax is None:
            n = len(result.performance.labels)  # type: ignore[union-attr]
            size = min(3.2 + 0.45 * n, 9)
            fig, ax = new_figure((size + 0.6, size), managed=True)
        else:
            fig = ax.figure
        _draw_confusion(ax, result, normalize)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------------------


def _draw_residuals_vs_pred(ax: Any, result: AnalysisResult) -> None:
    pred, r = result.y_pred, result.y_true - result.y_pred
    idx = np.arange(len(r))
    if len(idx) > 5000:
        idx = np.random.default_rng(0).choice(idx, 5000, replace=False)
    ax.scatter(pred[idx], r[idx], s=6, alpha=0.35, color=BLUE, linewidths=0, zorder=2)
    ax.axhline(0, color=BASELINE, linestyle="--", linewidth=1)
    if result.residuals and result.residuals.residual_by_prediction_decile:
        bins = result.residuals.residual_by_prediction_decile
        ax.plot([b["mean_predicted"] for b in bins], [b["mean_residual"] for b in bins],
                color=ORANGE, linewidth=2, marker="o", markersize=4, label="mean residual by "
                                                                            "prediction decile",
                zorder=3)
        ax.legend(loc="upper right")
    ax.set_xlabel("predicted")
    ax.set_ylabel("residual (actual − predicted)")
    ax.set_title("Residuals vs predicted")


def _draw_pred_vs_actual(ax: Any, result: AnalysisResult) -> None:
    idx = np.arange(len(result.y_true))
    if len(idx) > 5000:
        idx = np.random.default_rng(0).choice(idx, 5000, replace=False)
    ax.scatter(result.y_true[idx], result.y_pred[idx], s=6, alpha=0.35, color=BLUE,
               linewidths=0, zorder=2)
    lo = float(min(result.y_true.min(), result.y_pred.min()))
    hi = float(max(result.y_true.max(), result.y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linestyle="--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")


def plot_residuals(result: AnalysisResult) -> Any:
    """Residuals vs predicted values and predicted vs actual (regression)."""
    if result.is_classification:
        raise UnsupportedTaskError("plot_residuals() is only available for regression.")
    with _style():
        fig, axes = new_figure((11, 4), managed=True, ncols=2)
        _draw_residuals_vs_pred(axes[0], result)
        _draw_pred_vs_actual(axes[1], result)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Heatmap
# --------------------------------------------------------------------------------------


def _draw_heatmap(ax: Any, result: AnalysisResult, feature_x: str | None,
                  feature_y: str | None) -> None:
    grid = result.interaction_grid
    if feature_x is not None and feature_y is not None:
        metric = ((result.y_true != result.y_pred).astype(float) if result.is_classification
                  else np.abs(result.y_true - result.y_pred))
        grid = error_grid(result.X, metric, feature_x, feature_y, result.feature_infos,
                          metric_name="error rate" if result.is_classification else "MAE")
    if grid is None:
        _empty(ax, "No feature pair available for an error heatmap")
        return
    values = np.ma.masked_invalid(grid.values)
    im = ax.imshow(values, cmap=sequential_cmap(), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(grid.x_labels)))
    ax.set_xticklabels([textwrap.shorten(v, 12) for v in grid.x_labels], rotation=30,
                       ha="right", fontsize=7)
    ax.set_yticks(range(len(grid.y_labels)))
    ax.set_yticklabels([textwrap.shorten(v, 14) for v in grid.y_labels], fontsize=7)
    ax.set_xlabel(grid.feature_x)
    ax.set_ylabel(grid.feature_y)
    is_rate = result.is_classification
    vmax = float(np.nanmax(grid.values)) if np.isfinite(grid.values).any() else 1.0
    for i in range(len(grid.y_labels)):
        for j in range(len(grid.x_labels)):
            v = grid.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, _fmt_metric(v, is_rate), ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 * vmax else TEXT)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    if is_rate:
        cbar.ax.yaxis.set_major_formatter(mpl().ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{grid.metric_name.capitalize()} by {grid.feature_x} × {grid.feature_y}")


def plot_error_heatmap(result: AnalysisResult, feature_x: str | None = None,
                       feature_y: str | None = None, ax: Any = None) -> Any:
    """Two-feature heatmap of the error metric (cells with < 10 rows are blank)."""
    if (feature_x is None) != (feature_y is None):
        raise InvalidInputError("Pass both feature_x and feature_y, or neither.")
    with _style():
        if ax is None:
            fig, ax = new_figure((7, 4.8), managed=True)
        else:
            fig = ax.figure
        _draw_heatmap(ax, result, feature_x, feature_y)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------


def plot_overview(result: AnalysisResult) -> Any:
    """A 2×2 dashboard: performance view, patterns, feature errors and error heatmap."""
    with _style():
        fig, axes = new_figure((13, 9), managed=True, nrows=2, ncols=2)
        if result.is_classification:
            _draw_confusion(axes[0, 0], result, normalize=False)
        else:
            _draw_residuals_vs_pred(axes[0, 0], result)
        _draw_patterns(axes[0, 1], result.patterns, 6,
                       "Top failure patterns" if result.is_classification
                       else "Highest-error regions")
        _draw_feature_errors(axes[1, 0], result, _association(result, None))
        _draw_heatmap(axes[1, 1], result, None, None)
        fig.suptitle("ErrorLens overview", x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout()
    return fig
