"""Core data structures shared by the analysis, discovery and reporting layers.

Everything here is a plain dataclass with a ``to_dict()`` method so that the full analysis
can be serialized to JSON without custom encoders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from errorlens.utils.formatting import fmt_int, fmt_lift, fmt_num, fmt_p, fmt_pct

# --------------------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses / numpy / pandas values into JSON-safe objects."""
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(_scalar(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    return _scalar(value)


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _fmt_value(value: Any) -> str:
    """Compact, exact-looking rendering of a condition value (no trailing zeros)."""
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
            value, (bool, np.bool_)):
        v = float(value)
        if v.is_integer() and abs(v) < 1e15:
            return f"{v:,.0f}"
        return f"{v:,.6g}"
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value))
    return str(value)


# --------------------------------------------------------------------------------------
# Conditions and patterns
# --------------------------------------------------------------------------------------

EVENT_NOUNS = {
    "error": "errors",
    "false_positive": "false positives",
    "false_negative": "false negatives",
    "class_error": "missed predictions",
    "underprediction": "severe underpredictions",
    "overprediction": "severe overpredictions",
}

CONDITION_OPS = ("<=", ">", "==", "!=", "in_range", "is_missing", "not_missing")


@dataclass(frozen=True)
class Condition:
    """A single interpretable predicate on one feature.

    Missing values never satisfy a comparison (``<=``, ``>``, ``==``, ``!=``, ``in_range``);
    use ``is_missing`` to describe missingness explicitly.
    """

    feature: str
    op: str
    value: Any = None
    upper: Any = None

    def __post_init__(self) -> None:
        if self.op not in CONDITION_OPS:
            raise ValueError(f"Unsupported condition operator {self.op!r}")

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean membership of each row of ``X``."""
        if self.feature not in X.columns:
            raise KeyError(f"Feature {self.feature!r} not found in data")
        col = X[self.feature]
        notna = col.notna().to_numpy()
        if self.op == "is_missing":
            return ~notna
        if self.op == "not_missing":
            return notna
        if self.op in ("<=", ">", "in_range"):
            values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                if self.op == "<=":
                    out = values <= float(self.value)
                elif self.op == ">":
                    out = values > float(self.value)
                else:
                    out = (values > float(self.value)) & (values <= float(self.upper))
            return np.asarray(out & ~np.isnan(values), dtype=bool)
        eq = (col == self.value).to_numpy(dtype=bool, na_value=False)
        if self.op == "==":
            return eq & notna
        return (~eq) & notna

    @property
    def is_numeric(self) -> bool:
        return self.op in ("<=", ">", "in_range")

    def __str__(self) -> str:
        if self.op == "is_missing":
            return f"{self.feature} is missing"
        if self.op == "not_missing":
            return f"{self.feature} is not missing"
        if self.op == "in_range":
            return f"{_fmt_value(self.value)} < {self.feature} <= {_fmt_value(self.upper)}"
        return f"{self.feature} {self.op} {_fmt_value(self.value)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "op": self.op,
            "value": _scalar(self.value),
            "upper": _scalar(self.upper),
            "text": str(self),
        }


def conditions_mask(conditions: tuple[Condition, ...], X: pd.DataFrame) -> np.ndarray:
    """Conjunction (AND) of conditions evaluated on ``X``."""
    mask = np.ones(len(X), dtype=bool)
    for cond in conditions:
        mask &= cond.mask(X)
    return mask


def describe_conditions(conditions: tuple[Condition, ...], joiner: str = " AND ") -> str:
    return joiner.join(str(c) for c in conditions) if conditions else "(all rows)"


@dataclass
class GroupStats:
    """Statistics of one subgroup relative to its complement within a population.

    For binary targets ``value`` is an event rate (e.g. error rate) and ``n_events`` the
    event count. For continuous targets ``value`` is the mean (e.g. mean absolute error) and
    ``n_events`` is ``None``.
    """

    n: int
    value: float
    baseline: float
    complement_value: float
    lift: float
    coverage: float
    event_coverage: float
    population_size: int
    n_events: int | None = None
    relative_risk: float = float("nan")
    rr_ci_low: float = float("nan")
    rr_ci_high: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    effect_size: float = float("nan")
    effect_size_name: str = ""
    test: str = ""
    inference: str = "descriptive"

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}


@dataclass
class FailurePattern:
    """An interpretable subgroup with an unusually high error metric.

    ``stats`` are descriptive statistics on the full analysed population. ``validation``
    holds the inferential statistics (p-value, confidence interval, effect size), computed on
    an independent validation split when honest inference is used, otherwise in-sample.
    """

    conditions: tuple[Condition, ...]
    kind: str
    metric_name: str
    is_binary: bool
    stats: GroupStats
    validation: GroupStats
    p_value_adjusted: float = float("nan")
    significant: bool = False
    score: float = 0.0
    rank: int = 0
    mean_residual: float | None = None
    target_label: Any = None
    correction: str = ""

    # --- convenience accessors -------------------------------------------------------
    @property
    def complexity(self) -> int:
        return len(self.conditions)

    @property
    def description(self) -> str:
        return describe_conditions(self.conditions)

    @property
    def n_samples(self) -> int:
        return self.stats.n

    @property
    def n_errors(self) -> int | None:
        return self.stats.n_events

    @property
    def rate(self) -> float:
        """Subgroup metric value on the full data (error rate, FNR, MAE, ...)."""
        return self.stats.value

    @property
    def baseline(self) -> float:
        return self.stats.baseline

    @property
    def lift(self) -> float:
        return self.stats.lift

    @property
    def coverage(self) -> float:
        return self.stats.coverage

    @property
    def p_value(self) -> float:
        return self.validation.p_value

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return (self.validation.ci_low, self.validation.ci_high)

    @property
    def effect_size(self) -> float:
        return self.validation.effect_size

    @property
    def relative_risk(self) -> float:
        return self.validation.relative_risk

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the pattern to (new) data and return row membership."""
        return conditions_mask(self.conditions, X)

    def __str__(self) -> str:
        return self.description

    def headline(self) -> str:
        """One-sentence associational description of the pattern."""
        fmt = fmt_pct if self.is_binary else fmt_num
        return (f"{self.metric_name[0].upper() + self.metric_name[1:]} is {fmt(self.rate)} for "
                f"rows matching [{self.description}] vs {fmt(self.baseline)} overall "
                f"({fmt_lift(self.lift)} the baseline).")

    @property
    def event_noun(self) -> str:
        """What one 'event' is for this pattern's target (errors, false positives, ...)."""
        return EVENT_NOUNS.get(self.kind, "errors")

    def describe(self) -> str:
        """Multi-line, human-readable description (used by ``summary()``)."""
        v = self.validation
        label = "held-out" if v.inference == "holdout" else "in-sample"
        lines = ["  " + (f"AND {c}" if i else str(c)) for i, c in enumerate(self.conditions)]
        rows: list[tuple[str, str]] = [
            ("Samples", f"{fmt_int(self.n_samples)}  (coverage {fmt_pct(self.coverage, 2)})"),
        ]
        if self.is_binary:
            rows += [
                (self.event_noun.capitalize(), f"{fmt_int(self.n_errors)}  "
                 f"({fmt_pct(self.stats.event_coverage)} of all {self.event_noun})"),
                (self.metric_name.capitalize(), fmt_pct(self.rate, 2)),
                ("Baseline", fmt_pct(self.baseline, 2)),
            ]
            ci = f"[{fmt_pct(v.ci_low)}, {fmt_pct(v.ci_high)}]"
        else:
            rows += [(self.metric_name, fmt_num(self.rate)), ("Baseline", fmt_num(self.baseline))]
            if self.mean_residual is not None:
                direction = "under" if self.mean_residual > 0 else "over"
                rows.append(("Mean residual", f"{fmt_num(self.mean_residual)} "
                                              f"(tends to {direction}predict)"))
            ci = f"[{fmt_num(v.ci_low)}, {fmt_num(v.ci_high)}]"
        rows.append(("Lift", fmt_lift(self.lift)))
        rows.append((f"95% CI ({label})", ci))
        if self.is_binary and not math.isnan(v.relative_risk):
            rows.append(("Relative risk", f"{fmt_num(v.relative_risk)}  "
                                          f"[{fmt_num(v.rr_ci_low)}, {fmt_num(v.rr_ci_high)}]"
                                          " vs. rest"))
        adjusted = f"adjusted ({self.correction}): {fmt_p(self.p_value_adjusted)}"
        rows.append((f"p-value ({label})", f"{fmt_p(v.p_value)}   {adjusted}"))
        rows.append(("Effect size", f"{fmt_num(v.effect_size)} ({v.effect_size_name})"))
        rows.append(("Complexity", f"{self.complexity} condition"
                                   f"{'s' if self.complexity != 1 else ''}"))
        width = max(len(k) for k, _ in rows) + 2
        lines.append("")
        lines += [f"  {k + ':':<{width}}{val}" for k, val in rows]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "kind": self.kind,
            "target_label": _scalar(self.target_label),
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
            "complexity": self.complexity,
            "metric_name": self.metric_name,
            "is_binary": self.is_binary,
            "n_samples": self.n_samples,
            "n_errors": self.n_errors,
            "rate": _scalar(self.rate),
            "baseline": _scalar(self.baseline),
            "lift": _scalar(self.lift),
            "coverage": _scalar(self.coverage),
            "p_value": _scalar(self.p_value),
            "p_value_adjusted": _scalar(self.p_value_adjusted),
            "correction": self.correction,
            "significant": bool(self.significant),
            "score": _scalar(self.score),
            "mean_residual": _scalar(self.mean_residual),
            "stats": self.stats.to_dict(),
            "validation": self.validation.to_dict(),
        }


# --------------------------------------------------------------------------------------
# Data / performance summaries
# --------------------------------------------------------------------------------------


@dataclass
class FeatureInfo:
    name: str
    kind: str  # "numeric", "categorical", "boolean", "skipped"
    n_unique: int
    n_missing: int
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}


@dataclass
class DatasetSummary:
    n_samples: int
    n_features: int
    features: list[FeatureInfo]
    target_name: str
    n_missing_cells: int
    target_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def n_numeric(self) -> int:
        return sum(f.kind == "numeric" for f in self.features)

    @property
    def n_categorical(self) -> int:
        return sum(f.kind in ("categorical", "boolean") for f in self.features)

    @property
    def skipped(self) -> dict[str, str]:
        return {f.name: f.skipped_reason or "" for f in self.features if f.kind == "skipped"}

    @property
    def pct_missing(self) -> float:
        total = self.n_samples * max(self.n_features, 1)
        return self.n_missing_cells / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "n_missing_cells": self.n_missing_cells,
            "pct_missing": self.pct_missing,
            "target_name": self.target_name,
            "target_distribution": to_jsonable(self.target_distribution),
            "skipped_features": self.skipped,
            "features": [f.to_dict() for f in self.features],
        }


@dataclass
class ClassMetrics:
    label: Any
    precision: float
    recall: float
    f1: float
    support: int
    n_missed: int
    n_false_alarms: int

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}


@dataclass
class ClassificationPerformance:
    n_samples: int
    n_errors: int
    accuracy: float
    error_rate: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    labels: list[Any]
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetrics]
    is_binary: bool
    positive_label: Any = None
    tp: int | None = None
    tn: int | None = None
    fp: int | None = None
    fn: int | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    roc_auc: float | None = None
    log_loss: float | None = None

    task = "classification"

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}
        data["task"] = self.task
        return data


@dataclass
class RegressionPerformance:
    n_samples: int
    mae: float
    mse: float
    rmse: float
    median_ae: float
    r2: float
    max_error: float
    mean_residual: float
    mape: float | None = None

    task = "regression"

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}
        data["task"] = self.task
        return data


@dataclass
class ConfusionPair:
    true_label: Any
    predicted_label: Any
    count: int
    share_of_errors: float

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}


@dataclass
class ErrorSummary:
    """Task-dependent summary of how the model errs."""

    n_errors: int
    error_rate: float
    top_confusions: list[ConfusionPair] = field(default_factory=list)
    mean_confidence_correct: float | None = None
    mean_confidence_errors: float | None = None
    n_high_confidence_errors: int | None = None
    high_confidence_threshold: float | None = None
    n_underpredicted: int | None = None
    n_overpredicted: int | None = None
    severe_threshold: float | None = None
    n_severe_under: int | None = None
    n_severe_over: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}


@dataclass
class ResidualAnalysis:
    """Residual diagnostics for regression (residual = y_true - y_pred)."""

    mean_residual: float
    median_residual: float
    std_residual: float
    skewness: float
    excess_kurtosis: float
    pct_underpredicted: float
    pct_overpredicted: float
    mean_underprediction: float
    mean_overprediction: float
    severe_threshold: float
    n_severe_under: int
    n_severe_over: int
    heteroscedasticity_rho: float
    heteroscedasticity_p: float
    bias_test_p: float
    median_relative_error: float | None
    residual_by_prediction_decile: list[dict[str, Any]]
    feature_residual_correlations: list[dict[str, Any]]

    def interpretation(self) -> list[str]:
        notes = []
        if self.bias_test_p < 0.05:
            direction = "under" if self.mean_residual > 0 else "over"
            notes.append(
                f"Residuals have a non-zero mean ({fmt_num(self.mean_residual)}; "
                f"p {fmt_p(self.bias_test_p)}): the model tends to {direction}predict overall."
            )
        else:
            notes.append("No statistically significant overall bias in the residuals.")
        if self.heteroscedasticity_p < 0.05 and abs(self.heteroscedasticity_rho) >= 0.1:
            trend = "grows" if self.heteroscedasticity_rho > 0 else "shrinks"
            notes.append(
                f"Absolute error {trend} with the predicted value (Spearman rho = "
                f"{fmt_num(self.heteroscedasticity_rho)}), i.e. errors are heteroscedastic."
            )
        if abs(self.skewness) > 1:
            notes.append(
                f"The residual distribution is skewed (skewness {fmt_num(self.skewness)}).")
        if self.excess_kurtosis > 3:
            notes.append(
                f"Residuals are heavy-tailed (excess kurtosis {fmt_num(self.excess_kurtosis)}): "
                "a few predictions are far off."
            )
        return notes

    def __str__(self) -> str:
        lines = [
            "RESIDUAL ANALYSIS (residual = actual - predicted)",
            f"  Mean residual:      {fmt_num(self.mean_residual)}",
            f"  Median residual:    {fmt_num(self.median_residual)}",
            f"  Std of residuals:   {fmt_num(self.std_residual)}",
            f"  Underpredicted:     {fmt_pct(self.pct_underpredicted)} of rows "
            f"(mean {fmt_num(self.mean_underprediction)})",
            f"  Overpredicted:      {fmt_pct(self.pct_overpredicted)} of rows "
            f"(mean {fmt_num(self.mean_overprediction)})",
            f"  Severe threshold:   |residual| > {fmt_num(self.severe_threshold)} "
            f"({self.n_severe_under} under, {self.n_severe_over} over)",
        ]
        lines += ["  - " + n for n in self.interpretation()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        data = {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}
        data["interpretation"] = self.interpretation()
        return data


@dataclass
class FeatureAssociation:
    """Association between one feature and model errors (never a causal claim)."""

    feature: str
    feature_type: str
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_name: str
    magnitude: str
    p_value_adjusted: float = float("nan")
    summary_errors: str = ""
    summary_correct: str = ""
    direction: str = ""
    levels: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}


@dataclass
class DiscoveryStatistics:
    """Book-keeping for one discovery run — makes the multiple-testing family explicit."""

    kind: str
    metric_name: str
    population_size: int
    n_events: int | None
    baseline: float
    honest: bool
    n_discovery: int
    n_validation: int
    n_conditions: int
    n_candidates_evaluated: int
    n_candidates_tested: int
    family_size: int
    n_significant: int
    n_reported: int
    correction: str
    alpha: float
    target_label: Any = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _scalar(getattr(self, f.name)) for f in fields(self)}


@dataclass
class AnalysisMetadata:
    title: str
    created_at: str
    errorlens_version: str
    task: str
    model_info: dict[str, Any]
    config: dict[str, Any]
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}
