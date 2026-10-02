"""Plain-text console summary of an analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from errorlens.core.types import ClassificationPerformance, FailurePattern, RegressionPerformance
from errorlens.utils.formatting import fmt_int, fmt_lift, fmt_num, fmt_p, fmt_pct

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult

RULE = "─" * 60
KIND_TITLES = {
    "error": "FAILURE PATTERNS",
    "false_negative": "FALSE NEGATIVE PATTERNS",
    "false_positive": "FALSE POSITIVE PATTERNS",
    "class_error": "CLASS-SPECIFIC PATTERNS",
    "high_error": "HIGH-ERROR REGIONS",
    "underprediction": "SEVERE UNDERPREDICTION PATTERNS",
    "overprediction": "SEVERE OVERPREDICTION PATTERNS",
}


def _header(title: str) -> list[str]:
    return ["", title, RULE]


def render_summary(result: AnalysisResult, max_patterns: int = 5) -> str:
    lines: list[str] = ["ERRORLENS — ML MODEL ERROR ANALYSIS", "═" * 60]
    ds = result.dataset
    lines.append(f"Task: {result.task}   Samples: {fmt_int(ds.n_samples)}   "
                 f"Features: {ds.n_features}   Model: {result.metadata.model_info.get('type')}")
    perf = result.performance
    if isinstance(perf, ClassificationPerformance):
        lines += _classification_block(perf)
    elif isinstance(perf, RegressionPerformance):
        lines += _regression_block(result, perf)

    lines += _patterns_block(KIND_TITLES[result.primary_kind], result.patterns, max_patterns)
    for kind, pats in result.directional_patterns.items():
        lines += _patterns_block(KIND_TITLES[kind], pats, min(3, max_patterns))
    for label, pats in result.class_patterns_map.items():
        lines += _patterns_block(f"CLASS {label} — MISSED PREDICTIONS", pats, 2)

    if result.feature_analysis:
        lines += _header("FEATURE ERROR ASSOCIATION")
        lines.append(f"{'Feature':<22}{'Effect':<12}{'Size':>8}   {'Adj. p':<10}")
        for a in result.feature_analysis[:10]:
            lines.append(f"{a.feature[:21]:<22}{a.magnitude.capitalize():<12}"
                         f"{fmt_num(a.effect_size):>8}   {fmt_p(a.p_value_adjusted):<10}")
        lines.append("(effect: rank-biserial r / Cramér's V / Spearman rho / epsilon²)")

    if result.warnings:
        lines += _header("WARNINGS")
        lines += [f"• {w}" for w in result.warnings]
    lines.append("")
    lines.append("Patterns describe associations observed in this dataset, not causes of error.")
    return "\n".join(lines)


def _classification_block(perf: ClassificationPerformance) -> list[str]:
    lines = _header("CLASSIFICATION ERROR ANALYSIS")
    lines.append(f"Overall accuracy:   {fmt_pct(perf.accuracy)}")
    lines.append(f"Error rate:         {fmt_pct(perf.error_rate)}  "
                 f"({fmt_int(perf.n_errors)} of {fmt_int(perf.n_samples)})")
    lines.append(f"Balanced accuracy:  {fmt_pct(perf.balanced_accuracy)}")
    if perf.roc_auc is not None:
        lines.append(f"ROC AUC:            {fmt_num(perf.roc_auc)}")
    if perf.is_binary:
        lines.append(f"Positive class:     {perf.positive_label}")
        lines.append(f"False positives:    {fmt_int(perf.fp)}  "
                     f"(FPR {fmt_pct(perf.false_positive_rate)})")
        lines.append(f"False negatives:    {fmt_int(perf.fn)}  "
                     f"(FNR {fmt_pct(perf.false_negative_rate)})")
    for c in perf.per_class[:12]:
        lines.append("")
        lines.append(f"Class {c.label}")
        lines.append("─" * 8)
        lines.append(f"Precision: {fmt_pct(c.precision)}")
        lines.append(f"Recall:    {fmt_pct(c.recall)}")
        lines.append(f"Errors:    {fmt_int(c.n_missed)} missed, "
                     f"{fmt_int(c.n_false_alarms)} false alarms")
    if len(perf.per_class) > 12:
        lines.append(f"... {len(perf.per_class) - 12} more classes")
    return lines


def _regression_block(result: AnalysisResult, perf: RegressionPerformance) -> list[str]:
    lines = _header("REGRESSION ERROR ANALYSIS")
    lines.append(f"MAE:   {fmt_num(perf.mae)}")
    lines.append(f"RMSE:  {fmt_num(perf.rmse)}")
    lines.append(f"R²:    {fmt_num(perf.r2)}")
    lines.append(f"Mean residual (actual − predicted): {fmt_num(perf.mean_residual)}")
    if perf.mape is not None:
        lines.append(f"MAPE:  {fmt_pct(perf.mape)}")
    if result.residuals is not None:
        lines += ["  - " + n for n in result.residuals.interpretation()]
    return lines


def _patterns_block(title: str, patterns: list[FailurePattern], limit: int) -> list[str]:
    lines = _header(title)
    if not patterns:
        lines.append("No statistically significant pattern found.")
        return lines
    for p in patterns[:limit]:
        lines.append(f"Pattern #{p.rank}")
        lines.append(p.describe())
        lines.append("")
    if len(patterns) > limit:
        lines.append(f"... {len(patterns) - limit} more (see result.patterns_frame())")
    best = patterns[0]
    fmt = fmt_pct if best.is_binary else fmt_num
    lines.append(f"Top finding: {best.metric_name} {fmt(best.rate)} vs {fmt(best.baseline)} "
                 f"baseline ({fmt_lift(best.lift)}).")
    return lines
