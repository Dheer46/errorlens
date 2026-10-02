"""Model adapter: prediction, probability access and task inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from errorlens.exceptions import ModelError, PredictionError, UnsupportedTaskError

MAX_CLASSES = 200


class ModelAdapter:
    """Thin wrapper around any object with a scikit-learn style ``predict`` method."""

    def __init__(self, model: Any) -> None:
        if model is None:
            raise ModelError("model is None. Pass a fitted model or use "
                             "ErrorLens.from_predictions(X, y, y_pred).")
        if isinstance(model, (str, bytes)):
            raise ModelError("model must be a fitted model object, not a string. "
                             "Load it first (e.g. with joblib.load).")
        if not callable(getattr(model, "predict", None)):
            raise ModelError(
                f"Unsupported model type {type(model).__name__}: it has no callable "
                "'predict' method. ErrorLens works with any object exposing "
                "predict(X) (scikit-learn, XGBoost, LightGBM, CatBoost, Keras wrappers, ...)."
            )
        self.model = model

    @property
    def has_proba(self) -> bool:
        return callable(getattr(self.model, "predict_proba", None))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            pred = self.model.predict(X)
        except Exception as exc:
            raise PredictionError(
                f"{type(self.model).__name__}.predict(X) failed: {type(exc).__name__}: {exc}. "
                "Make sure X has the same columns and preprocessing the model was trained with."
            ) from exc
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred.ravel()
        if pred.ndim != 1:
            raise PredictionError(
                f"predict(X) returned an array with shape {pred.shape}; expected one prediction "
                "per row. Multi-output models are not supported."
            )
        if len(pred) != len(X):
            raise PredictionError(
                f"predict(X) returned {len(pred)} predictions for {len(X)} rows."
            )
        if pd.isna(pred).any():
            raise PredictionError("predict(X) returned missing (NaN) predictions.")
        return pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return class probabilities, or None if unavailable or failing."""
        if not self.has_proba:
            return None
        try:
            proba = np.asarray(self.model.predict_proba(X), dtype=float)
        except Exception:
            # Probabilities are optional context (confidence analysis); a model whose
            # predict_proba is unusable (e.g. SVC without probability=True) still works.
            return None
        if proba.ndim != 2 or len(proba) != len(X) or not np.isfinite(proba).all():
            return None
        return proba

    def classes(self) -> list[Any] | None:
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return None
        return list(np.asarray(classes).tolist())

    def info(self) -> dict[str, Any]:
        return model_info(self.model)


def model_info(model: Any) -> dict[str, Any]:
    """Collect non-sensitive descriptive information about a model."""
    if model is None:
        return {"type": "precomputed predictions"}
    cls = type(model)
    info: dict[str, Any] = {"type": cls.__name__, "module": cls.__module__}
    steps = getattr(model, "steps", None)
    if isinstance(steps, list):
        info["pipeline_steps"] = [f"{name}: {type(step).__name__}" for name, step in steps]
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        info["n_features_in"] = int(n_features)
    get_params = getattr(model, "get_params", None)
    if callable(get_params):
        try:
            params = get_params(deep=False)
            info["params"] = {
                k: v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__
                for k, v in params.items()
            }
        except Exception:
            pass
    return info


def infer_task(y: np.ndarray, model: Any = None, requested: str = "auto") -> str:
    """Decide between classification and regression."""
    if requested in ("classification", "regression"):
        if requested == "regression" and not np.issubdtype(np.asarray(y).dtype, np.number):
            raise UnsupportedTaskError("task='regression' requires a numeric target.")
        return requested
    if requested != "auto":
        raise UnsupportedTaskError(f"Unsupported task {requested!r}; "
                                   "use 'classification' or 'regression'.")
    if model is not None:
        estimator_type = getattr(model, "_estimator_type", None)
        try:  # scikit-learn >= 1.6 exposes tags
            from sklearn.utils import get_tags

            estimator_type = get_tags(model).estimator_type or estimator_type
        except Exception:
            pass
        if estimator_type == "classifier":
            return "classification"
        if estimator_type == "regressor":
            return "regression"
        if callable(getattr(model, "predict_proba", None)):
            return "classification"
    y = np.asarray(y)
    if y.dtype.kind in "OUSb" or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if y.dtype.kind == "f":
        finite = y[np.isfinite(y)]
        if len(finite) and not np.all(np.mod(finite, 1) == 0):
            return "regression"
    n_unique = len(np.unique(y))
    if n_unique <= 20 or n_unique <= 0.05 * len(y):
        return "classification"
    return "regression"


def validate_classification_target(y: np.ndarray, y_pred: np.ndarray) -> list[Any]:
    labels = sorted(set(pd.unique(y)) | set(pd.unique(y_pred)), key=_sort_key)
    if len(set(pd.unique(y))) < 2:
        # A single class in y is allowed (errors are still well defined) but metrics like
        # recall for absent classes are undefined; the caller emits a warning.
        pass
    if len(labels) > MAX_CLASSES:
        raise UnsupportedTaskError(
            f"Found {len(labels)} distinct labels; classification analysis supports at most "
            f"{MAX_CLASSES}. If the target is continuous, pass task='regression'."
        )
    return labels


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        return (0, float(value))
    return (1, str(value))
