"""Input validation and feature typing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from errorlens.core.config import AnalysisConfig
from errorlens.core.types import FeatureInfo
from errorlens.exceptions import InsufficientDataError, InvalidInputError

MIN_ROWS = 10


def to_frame(X: Any) -> pd.DataFrame:
    """Convert supported feature containers into a DataFrame with string column names."""
    if X is None:
        raise InvalidInputError("X is required (got None).")
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, pd.Series):
        df = X.to_frame()
    else:
        try:
            arr = np.asarray(X)
        except Exception as exc:
            raise InvalidInputError(f"Could not convert X of type {type(X).__name__} "
                                    "to an array.") from exc
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise InvalidInputError(
                f"X must be 2-dimensional, got an array with shape {arr.shape}.")
        df = pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])
    if df.shape[1] == 0:
        raise InvalidInputError("X has no columns.")
    df.columns = [str(c) for c in df.columns]
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise InvalidInputError(f"X has duplicate column names: {dupes}")
    return df.reset_index(drop=True)


def to_vector(y: Any, name: str, n_rows: int | None = None) -> np.ndarray:
    """Convert a target / prediction container into a 1-D numpy array and validate it."""
    if y is None:
        raise InvalidInputError(f"{name} is required (got None).")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise InvalidInputError(f"{name} must be 1-dimensional, got a DataFrame with "
                                    f"{y.shape[1]} columns.")
        y = y.iloc[:, 0]
    arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise InvalidInputError(f"{name} must be 1-dimensional, got shape {arr.shape}.")
    if n_rows is not None and len(arr) != n_rows:
        raise InvalidInputError(
            f"X and {name} have inconsistent lengths: X has {n_rows} rows, {name} has {len(arr)}."
        )
    if len(arr) and pd.isna(arr).any():
        raise InvalidInputError(f"{name} contains {int(pd.isna(arr).sum())} missing values; "
                                "drop or impute them before analysis.")
    return arr


def check_size(n_rows: int, config: AnalysisConfig) -> list[str]:
    """Validate the number of rows and return warnings for small datasets."""
    if n_rows == 0:
        raise InvalidInputError("The dataset is empty (0 rows).")
    if n_rows < MIN_ROWS:
        raise InsufficientDataError(
            f"At least {MIN_ROWS} rows are needed for error analysis, got {n_rows}."
        )
    warnings = []
    if n_rows < 2 * config.min_samples:
        warnings.append(
            f"Only {n_rows} rows with min_samples={config.min_samples}: no subgroup can be "
            "compared against a complement of comparable size, so pattern discovery is skipped. "
            "Lower min_samples or provide more data."
        )
    elif n_rows < 500:
        warnings.append(
            f"Small dataset ({n_rows} rows): statistical power is limited and only large "
            "effects can be detected."
        )
    return warnings


def infer_feature_types(X: pd.DataFrame, config: AnalysisConfig) -> list[FeatureInfo]:
    """Classify each column as numeric / categorical / boolean, or skip it with a reason."""
    infos: list[FeatureInfo] = []
    selected = set(config.features) if config.features is not None else None
    if selected is not None:
        unknown = selected - set(X.columns)
        if unknown:
            raise InvalidInputError(f"features not found in X: {sorted(unknown)}")
    ignored = set(config.ignore_features)
    for name in X.columns:
        col = X[name]
        n_missing = int(col.isna().sum())
        n_unique = int(col.nunique(dropna=True))
        n_present = len(col) - n_missing

        def skip(reason: str, *, _name: str = name, _u: int = n_unique,
                 _m: int = n_missing) -> FeatureInfo:
            return FeatureInfo(_name, "skipped", _u, _m, reason)

        if name in ignored:
            infos.append(skip("ignored by user"))
        elif selected is not None and name not in selected:
            infos.append(skip("not in selected features"))
        elif n_present == 0:
            infos.append(skip("all values missing"))
        elif pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_timedelta64_dtype(col):
            infos.append(skip("datetime features are not supported"))
        elif n_unique <= 1 and n_missing == 0:
            infos.append(skip("constant"))
        elif pd.api.types.is_bool_dtype(col):
            infos.append(FeatureInfo(name, "boolean", n_unique, n_missing))
        elif pd.api.types.is_numeric_dtype(col):
            infos.append(FeatureInfo(name, "numeric", n_unique, n_missing))
        elif _is_identifier_like(n_unique, n_present, config.max_categories):
            infos.append(skip(f"high-cardinality categorical ({n_unique} unique values); "
                              "looks like an identifier"))
        else:
            infos.append(FeatureInfo(name, "categorical", n_unique, n_missing))
    return infos


def _is_identifier_like(n_unique: int, n_present: int, max_categories: int) -> bool:
    return n_unique > max(max_categories, 50) and n_unique > 0.5 * n_present
