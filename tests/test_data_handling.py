import numpy as np
import pandas as pd
import pytest

from errorlens import (
    ErrorLens,
    InsufficientDataError,
    InvalidInputError,
    ModelError,
    PredictionError,
    UnsupportedTaskError,
)
from tests.conftest import FixedPredictor, make_binary_case, make_features


def test_numpy_input_and_column_names():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = (X[:, 0] > 0).astype(int)
    y_pred = np.where((X[:, 1] > 1) & (rng.random(1000) < 0.6), 1 - y, y)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert [f.name for f in result.dataset.features] == ["x0", "x1", "x2"]
    assert result.patterns and result.patterns[0].conditions[0].feature == "x1"


def test_feature_typing_constant_highcard_datetime():
    n = 1200
    X = make_features(n)
    X["constant"] = 7
    X["customer_id"] = [f"id{i}" for i in range(n)]
    X["signup"] = pd.date_range("2020-01-01", periods=n, freq="h")
    X["city"] = np.random.default_rng(0).choice([f"c{i}" for i in range(80)], n)
    X["all_missing"] = np.nan
    _, y, y_pred = make_binary_case(n)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    kinds = {f.name: f.kind for f in result.dataset.features}
    assert kinds["constant"] == "skipped"
    assert kinds["customer_id"] == "skipped"
    assert kinds["signup"] == "skipped"
    assert kinds["all_missing"] == "skipped"
    assert kinds["city"] == "categorical"  # high-cardinality but not identifier-like
    assert kinds["student"] == "boolean"
    assert any("customer_id" in w for w in result.warnings)


def test_missing_values_in_features_are_supported():
    X, y, y_pred = make_binary_case(3000)
    X = X.copy()
    X.loc[::7, "income"] = np.nan
    X.loc[::5, "region"] = None
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert result.dataset.n_missing_cells > 0
    levels = {lv["level"] for a in result.feature_analysis if a.feature == "region"
              for lv in a.levels}
    assert "(missing)" in levels


def test_small_dataset_runs_with_warning():
    X, y, y_pred = make_binary_case(40)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert any("discovery is skipped" in w or "Small dataset" in w for w in result.warnings)
    assert result.patterns == []


def test_too_few_rows():
    X, y, y_pred = make_binary_case(5)
    with pytest.raises(InsufficientDataError):
        ErrorLens.from_predictions(X, y, y_pred).analyze()


def test_empty_dataset():
    X = pd.DataFrame({"a": []})
    with pytest.raises(InvalidInputError):
        ErrorLens.from_predictions(X, [], []).analyze()


def test_dimension_mismatch():
    X = make_features(100)
    with pytest.raises(InvalidInputError, match="inconsistent lengths"):
        ErrorLens.from_predictions(X, np.zeros(99), np.zeros(100))
    with pytest.raises(InvalidInputError, match="1-dimensional"):
        ErrorLens.from_predictions(X, np.zeros((100, 2)), np.zeros(100))


def test_missing_inputs():
    X = make_features(100)
    with pytest.raises(InvalidInputError, match="y"):
        ErrorLens(FixedPredictor(np.zeros(100)), X, None)
    with pytest.raises(InvalidInputError, match="model"):
        ErrorLens(None, X, np.zeros(100))
    with pytest.raises(InvalidInputError):
        ErrorLens(FixedPredictor(np.zeros(100)), None, np.zeros(100))
    y = np.zeros(100, dtype=float)
    y[3] = np.nan
    with pytest.raises(InvalidInputError, match="missing"):
        ErrorLens.from_predictions(X, y, np.zeros(100))


def test_invalid_model():
    X = make_features(100)
    with pytest.raises(ModelError):
        ErrorLens(object(), X, np.zeros(100))
    with pytest.raises(ModelError):
        ErrorLens("model.pkl", X, np.zeros(100))


def test_prediction_failure_is_wrapped():
    class Broken:
        def predict(self, X):
            raise ValueError("feature names mismatch")

    X = make_features(100)
    with pytest.raises(PredictionError, match="feature names mismatch") as info:
        ErrorLens(Broken(), X, np.zeros(100)).analyze()
    assert isinstance(info.value.__cause__, ValueError)


def test_prediction_wrong_shape():
    class Wrong:
        def predict(self, X):
            return np.zeros(len(X) - 1)

    with pytest.raises(PredictionError):
        ErrorLens(Wrong(), make_features(100), np.zeros(100)).analyze()


def test_invalid_configuration():
    X = make_features(100)
    for kwargs in ({"task": "clustering"}, {"min_samples": 0}, {"significance_level": 1.5},
                   {"correction": "magic"}, {"max_patterns": 0}, {"honest": "maybe"},
                   {"not_an_option": 1}):
        with pytest.raises(InvalidInputError):
            ErrorLens.from_predictions(X, np.zeros(100), np.zeros(100), **kwargs)


def test_regression_requires_numeric_target():
    X = make_features(100)
    with pytest.raises(UnsupportedTaskError):
        ErrorLens.from_predictions(X, np.array(["a"] * 100), np.array(["a"] * 100),
                                   task="regression").analyze()


def test_feature_selection_and_ignore():
    X, y, y_pred = make_binary_case(3000)
    r = ErrorLens.from_predictions(X, y, y_pred, ignore_features=["income"]).analyze()
    assert all(c.feature != "income" for p in r.all_patterns() for c in p.conditions)
    r = ErrorLens.from_predictions(X, y, y_pred, features=["region", "student"]).analyze()
    assert all(c.feature in {"region", "student"} for p in r.all_patterns()
               for c in p.conditions)
    with pytest.raises(InvalidInputError):
        ErrorLens.from_predictions(X, y, y_pred, features=["nope"]).analyze()
