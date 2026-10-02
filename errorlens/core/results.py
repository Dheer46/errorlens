"""AnalysisResult: the single structured output of an ErrorLens analysis.

Every view of the analysis — the console summary, plots, JSON, Markdown, HTML and PDF —
is derived from this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from errorlens.core.types import (
    AnalysisMetadata,
    ClassificationPerformance,
    DatasetSummary,
    DiscoveryStatistics,
    ErrorSummary,
    FailurePattern,
    FeatureAssociation,
    FeatureInfo,
    RegressionPerformance,
    ResidualAnalysis,
    to_jsonable,
)
from errorlens.exceptions import UnsupportedTaskError
from errorlens.utils.formatting import safe_print

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from errorlens.discovery.interactions import ErrorGrid


@dataclass
class AnalysisResult:
    """Structured result of :meth:`errorlens.ErrorLens.analyze`.

    Attributes:
        metadata: Title, timestamp, version, task, model information and configuration.
        dataset: Dataset summary and feature typing.
        performance: Overall classification or regression metrics.
        error_summary: How the model errs (confusions, confidence, over/under-prediction).
        patterns: Ranked, statistically validated failure patterns for overall errors
            (classification) or absolute error (regression).
        directional_patterns: ``{"false_positive": [...], "false_negative": [...]}`` for
            binary classification, ``{"underprediction": [...], "overprediction": [...]}``
            for regression.
        class_patterns_map: Per-class miss patterns (multiclass).
        feature_analysis: Feature-level association with errors.
        residuals: Residual diagnostics (regression only).
        discovery_statistics: Book-keeping of every discovery run (tests, family size, ...).
        interaction_grid: Two-feature error grid for the most relevant feature pair.
        warnings: Data-quality and methodological warnings.
    """

    metadata: AnalysisMetadata
    dataset: DatasetSummary
    performance: ClassificationPerformance | RegressionPerformance
    error_summary: ErrorSummary
    patterns: list[FailurePattern]
    directional_patterns: dict[str, list[FailurePattern]] = field(default_factory=dict)
    class_patterns_map: dict[Any, list[FailurePattern]] = field(default_factory=dict)
    feature_analysis: list[FeatureAssociation] = field(default_factory=list)
    residuals: ResidualAnalysis | None = None
    discovery_statistics: list[DiscoveryStatistics] = field(default_factory=list)
    interaction_grid: ErrorGrid | None = None
    warnings: list[str] = field(default_factory=list)
    # Raw data kept for plotting and row-level accessors (never serialized).
    X: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    y_true: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    y_pred: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    y_proba: np.ndarray | None = field(default=None, repr=False)
    feature_infos: list[FeatureInfo] = field(default_factory=list, repr=False)
    _figure_cache: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ basics
    @property
    def task(self) -> str:
        return self.metadata.task

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def is_binary(self) -> bool:
        perf = self.performance
        return isinstance(perf, ClassificationPerformance) and perf.is_binary

    @property
    def primary_kind(self) -> str:
        return "error" if self.is_classification else "high_error"

    def __repr__(self) -> str:
        return (f"AnalysisResult(task={self.task!r}, n_samples={self.dataset.n_samples}, "
                f"n_patterns={len(self.patterns)}, warnings={len(self.warnings)})")

    def summary(self, max_patterns: int = 5, print_summary: bool = True) -> str:
        """Print (and return) a human-readable summary of the analysis."""
        from errorlens.core.summary import render_summary

        text = render_summary(self, max_patterns=max_patterns)
        if print_summary:
            safe_print(text)
        return text

    # ------------------------------------------------------------------ classification
    def _require_classification(self, what: str) -> ClassificationPerformance:
        if not isinstance(self.performance, ClassificationPerformance):
            raise UnsupportedTaskError(f"{what} is only available for classification "
                                       f"(this analysis is {self.task}).")
        return self.performance

    def _require_binary(self, what: str) -> ClassificationPerformance:
        perf = self._require_classification(what)
        if not perf.is_binary:
            raise UnsupportedTaskError(
                f"{what} is only defined for binary classification; for multiclass use "
                "result.class_patterns(label) and result.confusion_matrix().")
        return perf

    def error_rate(self) -> float:
        """Fraction of misclassified rows (classification)."""
        return self._require_classification("error_rate()").error_rate

    def accuracy(self) -> float:
        return self._require_classification("accuracy()").accuracy

    def confusion_matrix(self) -> pd.DataFrame:
        """Confusion matrix with true labels as rows and predicted labels as columns."""
        perf = self._require_classification("confusion_matrix()")
        labels = [str(v) for v in perf.labels]
        return pd.DataFrame(perf.confusion_matrix,
                            index=pd.Index(labels, name="true"),
                            columns=pd.Index(labels, name="predicted"))

    def _rows(self, mask: np.ndarray) -> pd.DataFrame:
        df = self.X.loc[mask].copy()
        df["y_true"] = self.y_true[mask]
        df["y_pred"] = self.y_pred[mask]
        return df

    def errors(self) -> pd.DataFrame:
        """Rows the model got wrong (classification) or all rows sorted by |error| (regression)."""
        if self.is_classification:
            return self._rows(self.y_true != self.y_pred)
        df = self._rows(np.ones(len(self.y_true), dtype=bool))
        df["residual"] = df["y_true"].astype(float) - df["y_pred"].astype(float)
        df["abs_error"] = df["residual"].abs()
        return df.sort_values("abs_error", ascending=False)

    def correct(self) -> pd.DataFrame:
        self._require_classification("correct()")
        return self._rows(self.y_true == self.y_pred)

    def false_positives(self) -> pd.DataFrame:
        """Rows predicted positive whose true label is negative (binary classification)."""
        pos = self._require_binary("false_positives()").positive_label
        return self._rows((self.y_pred == pos) & (self.y_true != pos))

    def false_negatives(self) -> pd.DataFrame:
        """Rows predicted negative whose true label is positive (binary classification)."""
        pos = self._require_binary("false_negatives()").positive_label
        return self._rows((self.y_pred != pos) & (self.y_true == pos))

    def false_positive_patterns(self) -> list[FailurePattern]:
        self._require_binary("false_positive_patterns()")
        return self.directional_patterns.get("false_positive", [])

    def false_negative_patterns(self) -> list[FailurePattern]:
        self._require_binary("false_negative_patterns()")
        return self.directional_patterns.get("false_negative", [])

    def class_patterns(self, label: Any = None) -> Any:
        """Per-class miss patterns (multiclass). Returns a dict, or a list for one ``label``."""
        self._require_classification("class_patterns()")
        if label is None:
            return dict(self.class_patterns_map)
        for key, value in self.class_patterns_map.items():
            if key == label or str(key) == str(label):
                return value
        return []

    # ------------------------------------------------------------------ regression
    def residual_analysis(self) -> ResidualAnalysis:
        if self.residuals is None:
            raise UnsupportedTaskError("residual_analysis() is only available for regression.")
        return self.residuals

    def high_error_regions(self) -> list[FailurePattern]:
        """Subgroups with significantly higher mean absolute error (regression)."""
        if self.is_classification:
            raise UnsupportedTaskError("high_error_regions() is for regression; use "
                                       "result.patterns for classification.")
        return self.patterns

    def underprediction_patterns(self) -> list[FailurePattern]:
        return self.directional_patterns.get("underprediction", [])

    def overprediction_patterns(self) -> list[FailurePattern]:
        return self.directional_patterns.get("overprediction", [])

    # ------------------------------------------------------------------ tables
    def all_patterns(self) -> list[FailurePattern]:
        out = list(self.patterns)
        for pats in self.directional_patterns.values():
            out += pats
        for pats in self.class_patterns_map.values():
            out += pats
        return out

    def patterns_frame(self, kind: str | None = None) -> pd.DataFrame:
        """All reported patterns as a DataFrame (optionally filtered by ``kind``)."""
        rows = []
        for p in self.all_patterns():
            if kind is not None and p.kind != kind:
                continue
            v = p.validation
            rows.append({
                "kind": p.kind, "class": p.target_label, "rank": p.rank,
                "pattern": p.description, "complexity": p.complexity,
                "n_samples": p.n_samples, "n_errors": p.n_errors, "coverage": p.coverage,
                "metric": p.metric_name, "rate": p.rate, "baseline": p.baseline,
                "lift": p.lift, "relative_risk": v.relative_risk,
                "ci_low": v.ci_low, "ci_high": v.ci_high, "p_value": v.p_value,
                "p_value_adjusted": p.p_value_adjusted, "effect_size": v.effect_size,
                "inference": v.inference, "score": p.score,
            })
        return pd.DataFrame(rows)

    def feature_associations(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": a.feature, "type": a.feature_type, "test": a.test,
            "effect_size": a.effect_size, "effect_measure": a.effect_size_name,
            "magnitude": a.magnitude, "p_value": a.p_value,
            "p_value_adjusted": a.p_value_adjusted, "direction": a.direction,
        } for a in self.feature_analysis])

    # ------------------------------------------------------------------ plotting
    def plot(self) -> Figure:
        """Dashboard figure with the most informative plots for this task."""
        from errorlens.visualization import plots

        return plots.plot_overview(self)

    def plot_error_distribution(self, feature: str | None = None) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_error_distribution(self, feature=feature)

    def plot_feature_errors(self, feature: str | None = None) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_feature_errors(self, feature=feature)

    def plot_patterns(self, kind: str | None = None, top: int = 10) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_patterns(self, kind=kind, top=top)

    def plot_residuals(self) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_residuals(self)

    def plot_confusion_matrix(self, normalize: bool = False) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_confusion_matrix(self, normalize=normalize)

    def plot_error_heatmap(self, feature_x: str | None = None,
                           feature_y: str | None = None) -> Figure:
        from errorlens.visualization import plots

        return plots.plot_error_heatmap(self, feature_x=feature_x, feature_y=feature_y)

    # ------------------------------------------------------------------ export
    def to_dict(self) -> dict[str, Any]:
        """Machine-readable representation of the full analysis (JSON-safe)."""
        return {
            "schema_version": "1.0",
            "metadata": self.metadata.to_dict(),
            "dataset": self.dataset.to_dict(),
            "performance": self.performance.to_dict(),
            "error_summary": self.error_summary.to_dict(),
            "patterns": [p.to_dict() for p in self.patterns],
            "directional_patterns": {k: [p.to_dict() for p in v]
                                     for k, v in self.directional_patterns.items()},
            "class_patterns": {str(k): [p.to_dict() for p in v]
                               for k, v in self.class_patterns_map.items()},
            "feature_analysis": [a.to_dict() for a in self.feature_analysis],
            "residual_analysis": self.residuals.to_dict() if self.residuals else None,
            "discovery_statistics": [s.to_dict() for s in self.discovery_statistics],
            "interaction_grid": (self.interaction_grid.to_dict()
                                 if self.interaction_grid is not None else None),
            "warnings": list(self.warnings),
            "methodology": to_jsonable(_methodology(self.metadata.config)),
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        from errorlens.reporting.json_export import write_json

        return write_json(self, path, indent=indent)

    def to_markdown(self, path: str | Path | None = None, include_figures: bool = True) -> str:
        from errorlens.reporting.markdown import write_markdown

        return write_markdown(self, path, include_figures=include_figures)

    def to_html(self, path: str | Path | None = None, include_figures: bool = True) -> str:
        from errorlens.reporting.html import write_html

        return write_html(self, path, include_figures=include_figures)

    def to_pdf(self, path: str | Path, include_figures: bool = True) -> Path:
        from errorlens.reporting.pdf import write_pdf

        return write_pdf(self, path, include_figures=include_figures)

    def export(self, path: str | Path, format: str | None = None, **kwargs: Any) -> Path:
        """Export the report; the format is inferred from the extension unless given."""
        from errorlens.reporting.export import export

        return export(self, path, format=format, **kwargs)


def _methodology(config: dict[str, Any]) -> dict[str, str]:
    from errorlens.reporting.report import methodology_text

    return dict(methodology_text(config))
