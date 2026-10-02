import numpy as np
import pytest
from sklearn import metrics as skm

from errorlens import ErrorLens, UnsupportedTaskError
from errorlens.discovery.deduplication import jaccard
from tests.conftest import make_features


def test_metrics_match_sklearn(regression_case, regression_result):
    _, y, y_pred = regression_case
    perf = regression_result.performance
    assert perf.mae == pytest.approx(skm.mean_absolute_error(y, y_pred))
    assert perf.rmse == pytest.approx(np.sqrt(skm.mean_squared_error(y, y_pred)))
    assert perf.r2 == pytest.approx(skm.r2_score(y, y_pred))
    assert perf.mape is not None  # target bounded away from zero


def test_high_error_subgroup_found(regression_case, regression_result):
    X, _, _ = regression_case
    regions = regression_result.high_error_regions()
    assert regions
    top = regions[0]
    assert {c.feature for c in top.conditions} == {"age", "income"}
    truth = ((X["income"] > 100000) & (X["age"] < 30)).to_numpy()
    assert jaccard(top.mask(X), truth) > 0.7
    assert top.lift > 2.5
    assert top.metric_name == "MAE"
    assert not top.is_binary
    assert top.validation.test == "one-sided Welch t-test"


def test_normal_residuals_have_no_patterns():
    n = 6000
    X = make_features(n, 50)
    y_pred = np.random.default_rng(50).normal(100, 10, n)
    y = y_pred + np.random.default_rng(51).normal(0, 3, n)
    result = ErrorLens.from_predictions(X, y, y_pred, task="regression").analyze()
    assert result.patterns == []
    ra = result.residual_analysis()
    assert ra.bias_test_p > 0.01
    assert abs(ra.skewness) < 0.3


@pytest.mark.parametrize(("shift", "word"), [(-5.0, "under"), (5.0, "over")])
def test_systematic_bias(shift, word):
    n = 3000
    rng = np.random.default_rng(60)
    X = make_features(n, 60)
    y = rng.normal(100, 10, n)
    y_pred = y + shift + rng.normal(0, 1, n)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert result.task == "regression"
    ra = result.residual_analysis()
    assert ra.bias_test_p < 1e-6
    assert ra.mean_residual == pytest.approx(-shift, abs=0.2)
    assert any(f"{word}predict" in note for note in ra.interpretation())
    assert ("Underpredicted" in str(ra))


def test_directional_regression_patterns():
    n = 8000
    rng = np.random.default_rng(70)
    X = make_features(n, 70)
    y = rng.normal(100, 5, n)
    under = (X["region"] == "west").to_numpy() & (X["account_age"] > 15).to_numpy()
    y_pred = y + rng.normal(0, 1, n) - np.where(under, 8.0, 0.0)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    up = result.underprediction_patterns()
    assert up and {c.feature for c in up[0].conditions} == {"region", "account_age"}
    assert up[0].metric_name == "severe underprediction rate"
    assert result.high_error_regions()[0].mean_residual > 0  # model underpredicts there


def test_feature_residual_correlations(regression_result):
    rows = regression_result.residuals.feature_residual_correlations
    assert rows and {"feature", "measure", "value", "p_value_adjusted"} <= set(rows[0])
    assert regression_result.residuals.residual_by_prediction_decile


def test_regression_accessor_errors(regression_result):
    with pytest.raises(UnsupportedTaskError):
        regression_result.error_rate()
    with pytest.raises(UnsupportedTaskError):
        regression_result.confusion_matrix()
    errs = regression_result.errors()
    assert errs["abs_error"].is_monotonic_decreasing


def test_task_inference():
    X = make_features(400)
    y = np.random.default_rng(0).normal(size=400)
    r = ErrorLens.from_predictions(X, y, y + 0.1).analyze()
    assert r.task == "regression"
    y_int = np.random.default_rng(0).integers(0, 3, 400)
    r = ErrorLens.from_predictions(X, y_int, y_int).analyze()
    assert r.task == "classification"
