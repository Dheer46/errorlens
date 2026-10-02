"""Render the report's figure set to PNG bytes (no pyplot, no global state)."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from errorlens.utils.optional import is_available
from errorlens.visualization import plots
from errorlens.visualization._mpl import new_figure

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult

DPI = 160


@dataclass
class ReportFigure:
    key: str
    caption: str
    png: bytes
    width_in: float
    height_in: float


def matplotlib_available() -> bool:
    return is_available("matplotlib")


def _render(key: str, caption: str, size: tuple[float, float],
            draw: Callable[[Any], None], ncols: int = 1) -> ReportFigure:
    with plots._style():
        fig, axes = new_figure(size, managed=False, ncols=ncols)
        draw(axes)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI)
    return ReportFigure(key, caption, buf.getvalue(), size[0], size[1])


def _pattern_height(n: int) -> float:
    return 1.2 + 0.55 * max(min(n, 10), 2)


def render_report_figures(result: AnalysisResult) -> dict[str, ReportFigure]:
    """All figures used by HTML / PDF / Markdown reports, keyed by name."""
    figs: dict[str, ReportFigure] = {}
    if result.is_classification:
        n = len(result.performance.labels)  # type: ignore[union-attr]
        if n <= 25:
            side = min(3.4 + 0.4 * n, 7.5)
            figs["confusion"] = _render(
                "confusion", "Confusion matrix (rows: actual class, columns: predicted class).",
                (side + 0.8, side), lambda ax: plots._draw_confusion(ax, result, False))
    else:
        def draw_res(axes: Any) -> None:
            plots._draw_residuals_vs_pred(axes[0], result)
            plots._draw_pred_vs_actual(axes[1], result)

        figs["residuals"] = _render(
            "residuals", "Left: residuals against predictions with the mean residual per "
            "prediction decile. Right: predicted vs actual values (dashed: perfect prediction).",
            (10, 3.8), draw_res, ncols=2)
        figs["residual_hist"] = _render(
            "residual_hist", "Distribution of residuals (actual − predicted).", (7, 3.2),
            lambda ax: plots._draw_error_distribution(ax, result, None))

    titles = {"error": "Top failure patterns", "high_error": "Highest-error regions",
              "false_positive": "False-positive patterns",
              "false_negative": "False-negative patterns",
              "underprediction": "Severe underprediction patterns",
              "overprediction": "Severe overprediction patterns"}
    groups = {result.primary_kind: result.patterns, **result.directional_patterns}
    for kind, pats in groups.items():
        if not pats:
            continue

        def draw_patterns(ax: Any, p: list[Any] = pats, t: str = titles[kind]) -> None:
            plots._draw_patterns(ax, p, 10, t)

        figs[f"patterns_{kind}"] = _render(
            f"patterns_{kind}",
            f"{titles[kind]}: subgroup {pats[0].metric_name} on the full data (bars) with "
            "95% confidence intervals (whiskers) and the population baseline (dashed). "
            "Labels show the lift over baseline.",
            (8.0, _pattern_height(len(pats))), draw_patterns)

    for i, assoc in enumerate(result.feature_analysis[:4]):

        def draw_feature(ax: Any, a: Any = assoc) -> None:
            plots._draw_feature_errors(ax, result, a)

        figs[f"feature_{i}"] = _render(
            f"feature_{i}",
            f"{'Error rate' if result.is_classification else 'MAE'} across "
            f"{'levels' if assoc.feature_type != 'numeric' else 'quantile bins'} of "
            f"'{assoc.feature}' (dashed: overall).",
            (7, 3.2), draw_feature)

    if result.is_classification and result.feature_analysis:
        figs["distribution"] = _render(
            "distribution", "Distribution of the feature most associated with errors, for "
            "misclassified vs correctly classified rows.", (7, 3.2),
            lambda ax: plots._draw_error_distribution(ax, result, None))

    if result.interaction_grid is not None:
        g = result.interaction_grid
        figs["heatmap"] = _render(
            "heatmap", f"{g.metric_name.capitalize()} for combinations of '{g.feature_x}' and "
            f"'{g.feature_y}' (blank cells: fewer than {g.min_count} rows).", (7.5, 4.6),
            lambda ax: plots._draw_heatmap(ax, result, None, None))
    return figs
