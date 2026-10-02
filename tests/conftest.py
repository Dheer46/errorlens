"""Shared synthetic datasets with *known* failure patterns."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


class FixedPredictor:
    """A 'model' that returns precomputed predictions (lets tests control errors exactly)."""

    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = np.asarray(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predictions[: len(X)]


def make_features(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.lognormal(10.6, 0.55, n).round(0),
        "student": rng.random(n) < 0.2,
        "region": rng.choice(["north", "south", "east", "west"], n),
        "transactions": rng.poisson(10, n).astype(float),
        "account_age": rng.uniform(0, 20, n).round(1),
    })
    X.loc[rng.random(n) < 0.05, "transactions"] = np.nan
    return X


def planted_subgroup(X: pd.DataFrame) -> np.ndarray:
    return ((X["age"] < 25) & (X["income"] < 30000)).to_numpy()


def make_binary_case(n: int = 8000, seed: int = 0, base_error: float = 0.05,
                     subgroup_error: float = 0.40) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Binary labels with predictions that err at ``subgroup_error`` inside the planted group."""
    rng = np.random.default_rng(seed + 1)
    X = make_features(n, seed)
    y = (rng.random(n) < 0.4).astype(int)
    p_err = np.where(planted_subgroup(X), subgroup_error, base_error)
    flip = rng.random(n) < p_err
    y_pred = np.where(flip, 1 - y, y)
    return X, y, y_pred


def make_regression_case(n: int = 8000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray,
                                                                np.ndarray]:
    """Regression where rows with income > 100k and age < 30 have 6x larger noise."""
    rng = np.random.default_rng(seed + 2)
    X = make_features(n, seed)
    X["income"] = rng.uniform(20000, 160000, n).round(0)
    y_hat = 50 + 0.0003 * X["income"].to_numpy() + 0.4 * X["age"].to_numpy()
    hard = ((X["income"] > 100000) & (X["age"] < 30)).to_numpy()
    noise = rng.normal(0, np.where(hard, 12.0, 2.0))
    return X, y_hat + noise, y_hat


@pytest.fixture(scope="session")
def binary_case() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return make_binary_case()


@pytest.fixture(scope="session")
def binary_result(binary_case):  # type: ignore[no-untyped-def]
    from errorlens import ErrorLens

    X, y, y_pred = binary_case
    return ErrorLens(FixedPredictor(y_pred), X, y).analyze()


@pytest.fixture(scope="session")
def regression_case() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return make_regression_case()


@pytest.fixture(scope="session")
def regression_result(regression_case):  # type: ignore[no-untyped-def]
    from errorlens import ErrorLens

    X, y, y_pred = regression_case
    return ErrorLens.from_predictions(X, y, y_pred, task="regression").analyze()


@pytest.fixture(scope="session")
def multiclass_result():  # type: ignore[no-untyped-def]
    from errorlens import ErrorLens

    n = 6000
    rng = np.random.default_rng(5)
    X = make_features(n, 5)
    y = rng.choice(["cat", "dog", "bird"], n, p=[0.4, 0.4, 0.2])
    y_pred = y.copy()
    # Birds in the north are confused with dogs 50% of the time, else 4% random error.
    hard = (y == "bird") & (X["region"] == "north").to_numpy()
    flip = rng.random(n) < np.where(hard, 0.5, 0.04)
    y_pred[flip & hard] = "dog"
    others = flip & ~hard
    y_pred[others] = np.where(y[others] == "cat", "dog", "cat")
    return ErrorLens.from_predictions(X, y, y_pred).analyze()
