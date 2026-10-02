"""The ErrorLens entry point."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from errorlens._version import __version__
from errorlens.analysis.classification import (
    classification_error_summary,
    classification_performance,
    default_positive_label,
)
from errorlens.analysis.distributions import feature_associations
from errorlens.analysis.errors import classification_targets, regression_targets
from errorlens.analysis.regression import (
    regression_error_summary,
    regression_performance,
    residual_analysis,
)
from errorlens.core.config import AnalysisConfig
from errorlens.core.data import check_size, infer_feature_types, to_frame, to_vector
from errorlens.core.models import (
    ModelAdapter,
    infer_task,
    model_info,
    validate_classification_target,
)
from errorlens.core.results import AnalysisResult
from errorlens.core.types import AnalysisMetadata, DatasetSummary, FailurePattern
from errorlens.discovery.interactions import error_grid, top_feature_pair
from errorlens.discovery.subgroup import DiscoveryTarget, SubgroupDiscovery
from errorlens.exceptions import InvalidInputError, UnsupportedTaskError


class ErrorLens:
    """Automatic failure-pattern discovery for tabular classification and regression models.

    Example:
        >>> lens = ErrorLens(model, X_test, y_test)          # doctest: +SKIP
        >>> result = lens.analyze()                           # doctest: +SKIP
        >>> result.summary()                                  # doctest: +SKIP

    Args:
        model: Fitted model with a ``predict`` method (optionally ``predict_proba``). May be
            ``None`` if ``y_pred`` is given.
        X: Evaluation features (DataFrame or array). Should be data the model was *not*
            trained on.
        y: True targets for ``X``.
        y_pred: Precomputed predictions (alternative to ``model``).
        y_proba: Precomputed class probabilities (optional, classification only).
        config: An :class:`AnalysisConfig`; individual keyword arguments override it.
        title: Report title.
        **kwargs: Any :class:`AnalysisConfig` field, e.g. ``task``, ``max_patterns``,
            ``min_samples``, ``significance_level``, ``correction``, ``max_depth``.
    """

    def __init__(self, model: Any = None, X: Any = None, y: Any = None, *,
                 y_pred: Any = None, y_proba: Any = None,
                 config: AnalysisConfig | None = None, title: str = "ErrorLens Error Analysis",
                 **kwargs: Any) -> None:
        if config is None:
            try:
                config = AnalysisConfig(**kwargs)
            except TypeError as exc:
                raise InvalidInputError(f"Unknown ErrorLens option: {exc}") from exc
        elif kwargs:
            merged = {**config.to_dict(), **kwargs}
            merged["positive_label"] = kwargs.get("positive_label", config.positive_label)
            config = AnalysisConfig(**merged)
        self.config = config
        self.title = title

        if model is None and y_pred is None:
            raise InvalidInputError("Provide either a fitted model or precomputed y_pred.")
        if X is None:
            raise InvalidInputError("X (evaluation features) is required.")
        if y is None:
            raise InvalidInputError("y (true targets) is required — ErrorLens needs labels to "
                                    "identify errors.")
        self.X = to_frame(X)
        name = getattr(y, "name", None)
        self.target_name = str(name) if isinstance(name, (str, int)) else "target"
        self.y = to_vector(y, "y", len(self.X))
        self.adapter = ModelAdapter(model) if model is not None else None
        self.model = model
        self.y_pred = None if y_pred is None else to_vector(y_pred, "y_pred", len(self.X))
        self.y_proba = None
        if y_proba is not None:
            proba = np.asarray(y_proba, dtype=float)
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            if proba.ndim != 2 or len(proba) != len(self.X):
                raise InvalidInputError("y_proba must have shape (n_samples, n_classes).")
            self.y_proba = proba

    @classmethod
    def from_predictions(cls, X: Any, y: Any, y_pred: Any, y_proba: Any = None,
                         **kwargs: Any) -> ErrorLens:
        """Analyze precomputed predictions without a model object."""
        return cls(None, X, y, y_pred=y_pred, y_proba=y_proba, **kwargs)

    # ------------------------------------------------------------------------------
    def analyze(self) -> AnalysisResult:
        """Run the full analysis and return an :class:`AnalysisResult`."""
        start = time.perf_counter()
        cfg = self.config
        X, y = self.X, self.y
        warnings = check_size(len(X), cfg)
        infos = infer_feature_types(X, cfg)
        for info in infos:
            if info.kind == "skipped" and info.skipped_reason not in (
                    "ignored by user", "not in selected features"):
                warnings.append(f"Feature '{info.name}' was excluded: {info.skipped_reason}.")

        if self.y_pred is not None:
            y_pred = self.y_pred
            proba = self.y_proba
            proba_classes = None
        else:
            assert self.adapter is not None
            y_pred = self.adapter.predict(X)
            proba = self.adapter.predict_proba(X)
            proba_classes = self.adapter.classes()

        task = infer_task(y, self.model, cfg.task)
        if task == "classification":
            result = self._classification(X, y, y_pred, proba, proba_classes, infos, warnings)
        else:
            result = self._regression(X, y, y_pred, infos, warnings)
        result.metadata.runtime_seconds = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------------------
    def _metadata(self, task: str) -> AnalysisMetadata:
        info = self.adapter.info() if self.adapter is not None else model_info(None)
        return AnalysisMetadata(
            title=self.title,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            errorlens_version=__version__,
            task=task,
            model_info=info,
            config=self.config.to_dict(),
        )

    def _dataset(self, infos: list[Any], target_distribution: dict[str, int]) -> DatasetSummary:
        return DatasetSummary(
            n_samples=len(self.X),
            n_features=self.X.shape[1],
            features=infos,
            target_name=self.target_name,
            n_missing_cells=int(self.X.isna().sum().sum()),
            target_distribution=target_distribution,
        )

    def _discover(self, targets: list[DiscoveryTarget], infos: list[Any],
                  warnings: list[str]) -> tuple[dict[tuple[str, Any], list[FailurePattern]],
                                                list[Any]]:
        engine = SubgroupDiscovery(self.config, infos)
        found: dict[tuple[str, Any], list[FailurePattern]] = {}
        stats = []
        exploratory = False
        for target in targets:
            patterns, st = engine.run(self.X, target)
            found[(target.kind, target.target_label)] = patterns
            stats.append(st)
            if not st.honest and st.n_candidates_tested > 0:
                exploratory = True
            if target.kind in ("error", "high_error") and st.note and st.n_candidates_tested == 0:
                warnings.append(f"Pattern discovery: {st.note}")
        if exploratory:
            warnings.append(
                "Some patterns were tested on the same data used to discover them (dataset too "
                "small for a held-out validation split). Their p-values are optimistic; treat "
                "them as exploratory and confirm on fresh data.")
        return found, stats

    # ------------------------------------------------------------------------------
    def _classification(self, X: pd.DataFrame, y: np.ndarray, y_pred: np.ndarray,
                        proba: np.ndarray | None, proba_classes: list[Any] | None,
                        infos: list[Any], warnings: list[str]) -> AnalysisResult:
        cfg = self.config
        if y_pred.dtype.kind == "f" and y.dtype.kind in "iub":
            if np.all(np.mod(y_pred, 1) == 0):
                y_pred = y_pred.astype(y.dtype) if y.dtype.kind != "b" else y_pred.astype(bool)
            else:
                raise UnsupportedTaskError(
                    "Predictions are continuous but the task is classification. If this is a "
                    "regression model pass task='regression'; if predict() returns scores, "
                    "threshold them first.")
        labels = validate_classification_target(y, y_pred)
        true_labels = set(pd.unique(y))
        if not true_labels & set(pd.unique(y_pred)):
            warnings.append("No predicted label matches any true label — check that y and the "
                            "model's predictions use the same label encoding.")
        if len(true_labels) < 2:
            warnings.append("y contains a single class; per-class recall and ROC metrics are "
                            "limited.")
        positive = cfg.positive_label
        if len(labels) == 2:
            if positive is None:
                positive = default_positive_label(labels)
            elif positive not in labels:
                raise InvalidInputError(f"positive_label={positive!r} is not one of the labels "
                                        f"{labels}.")
        if proba_classes is None and proba is not None and proba.shape[1] == len(labels):
            proba_classes = labels
        perf = classification_performance(y, y_pred, labels, positive_label=positive,
                                          proba=proba, proba_classes=proba_classes)
        err_summary = classification_error_summary(y, y_pred, proba)
        errors = y != y_pred
        if not errors.any():
            warnings.append("The model made no errors on this data; there are no failure "
                            "patterns to discover.")
        elif errors.all():
            warnings.append("Every prediction is wrong; check label encoding and that the model "
                            "matches the data.")
        elif perf.error_rate > 0.5:
            warnings.append(f"Very high error rate ({perf.error_rate:.1%}); patterns describe "
                            "where the model is even worse than this baseline.")
        if perf.n_errors and perf.n_errors < 10:
            warnings.append(f"Only {perf.n_errors} errors: statistical power for pattern "
                            "discovery is very low.")

        targets = classification_targets(
            y, y_pred, labels, positive_label=positive, directional=cfg.analyze_directional,
            max_classes=cfg.max_classes, min_samples=cfg.min_samples)
        found, stats = self._discover(targets, infos, warnings)
        patterns = found.get(("error", None), [])
        directional = {kind: found[(kind, positive)] for kind in ("false_positive",
                                                                  "false_negative")
                       if (kind, positive) in found}
        class_map = {label: pats for (kind, label), pats in found.items()
                     if kind == "class_error"}

        assoc = (feature_associations(X, infos, errors=errors, abs_error=None)
                 if 0 < errors.sum() < len(errors) else [])
        grid = self._grid(patterns, assoc, errors.astype(float), "error rate", infos)
        dist = {str(k): int(v) for k, v in pd.Series(y).value_counts().items()}
        return AnalysisResult(
            metadata=self._metadata("classification"), dataset=self._dataset(infos, dist),
            performance=perf, error_summary=err_summary, patterns=patterns,
            directional_patterns=directional, class_patterns_map=class_map,
            feature_analysis=assoc, discovery_statistics=stats, interaction_grid=grid,
            warnings=warnings, X=X, y_true=y, y_pred=y_pred, y_proba=proba,
            feature_infos=infos,
        )

    def _regression(self, X: pd.DataFrame, y: np.ndarray, y_pred: np.ndarray,
                    infos: list[Any], warnings: list[str]) -> AnalysisResult:
        cfg = self.config
        try:
            y_f = np.asarray(y, dtype=float)
            p_f = np.asarray(y_pred, dtype=float)
        except (TypeError, ValueError) as exc:
            raise UnsupportedTaskError("Regression requires numeric y and predictions.") from exc
        if not (np.isfinite(y_f).all() and np.isfinite(p_f).all()):
            raise InvalidInputError("y and predictions must be finite for regression.")
        perf = regression_performance(y_f, p_f)
        targets, tau = regression_targets(y_f, p_f, directional=cfg.analyze_directional)
        err_summary = regression_error_summary(y_f, p_f, tau)
        residuals = residual_analysis(X, y_f, p_f, infos, tau)
        if perf.mae == 0:
            warnings.append("Predictions match the targets exactly; there are no errors to "
                            "analyze.")
        found, stats = self._discover(targets, infos, warnings)
        patterns = found.get(("high_error", None), [])
        directional = {kind: found[(kind, None)] for kind in ("underprediction",
                                                              "overprediction")
                       if (kind, None) in found}
        abs_err = np.abs(y_f - p_f)
        assoc = (feature_associations(X, infos, errors=None, abs_error=abs_err)
                 if perf.mae > 0 else [])
        grid = self._grid(patterns, assoc, abs_err, "MAE", infos)
        return AnalysisResult(
            metadata=self._metadata("regression"), dataset=self._dataset(infos, {}),
            performance=perf, error_summary=err_summary, patterns=patterns,
            directional_patterns=directional, feature_analysis=assoc, residuals=residuals,
            discovery_statistics=stats, interaction_grid=grid, warnings=warnings,
            X=X, y_true=y_f, y_pred=p_f, feature_infos=infos,
        )

    def _grid(self, patterns: list[FailurePattern], assoc: list[Any], metric: np.ndarray,
              metric_name: str, infos: list[Any]) -> Any:
        fallback = [a.feature for a in assoc]
        pair = top_feature_pair(patterns, fallback)
        if pair is None or len(self.X) < 50:
            return None
        return error_grid(self.X, metric, pair[0], pair[1], infos, metric_name=metric_name)
