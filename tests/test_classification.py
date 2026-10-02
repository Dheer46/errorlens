import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from errorlens import ErrorLens, InvalidInputError, UnsupportedTaskError
from errorlens.discovery.deduplication import jaccard
from tests.conftest import FixedPredictor, make_features


def test_binary_metrics_match_sklearn(binary_case, binary_result):
    _, y, y_pred = binary_case
    perf = binary_result.performance
    assert perf.is_binary
    assert perf.accuracy == pytest.approx(skm.accuracy_score(y, y_pred))
    assert binary_result.error_rate() == pytest.approx(1 - skm.accuracy_score(y, y_pred))
    tn, fp, fn, tp = skm.confusion_matrix(y, y_pred).ravel()
    assert (perf.tp, perf.tn, perf.fp, perf.fn) == (tp, tn, fp, fn)
    cm = binary_result.confusion_matrix()
    assert cm.loc["1", "0"] == fn and cm.loc["0", "1"] == fp
    assert perf.positive_label == 1


def test_false_positive_and_negative_rows(binary_case, binary_result):
    _, y, y_pred = binary_case
    fp = binary_result.false_positives()
    fn = binary_result.false_negatives()
    assert len(fp) == int(((y_pred == 1) & (y == 0)).sum())
    assert len(fn) == int(((y_pred == 0) & (y == 1)).sum())
    assert (fp["y_true"] == 0).all() and (fp["y_pred"] == 1).all()
    assert len(binary_result.errors()) == int((y != y_pred).sum())
    assert len(binary_result.correct()) + len(binary_result.errors()) == len(y)


def test_directional_patterns_found(binary_result):
    # errors are planted symmetrically, so both FP and FN analyses see the subgroup
    for pats in (binary_result.false_positive_patterns(),
                 binary_result.false_negative_patterns()):
        assert pats, "expected a directional pattern"
        assert {c.feature for c in pats[0].conditions} >= {"age"}
        assert pats[0].lift > 2


def test_false_negative_specific_pattern():
    """Only positives with many transactions are missed: FN patterns find it, FP do not."""
    n = 8000
    rng = np.random.default_rng(21)
    X = make_features(n, 21)
    y = (rng.random(n) < 0.5).astype(int)
    tx = X["transactions"].fillna(0).to_numpy()
    miss = (y == 1) & (rng.random(n) < np.where(tx > 13, 0.45, 0.04))
    false_alarm = (y == 0) & (rng.random(n) < 0.04)
    y_pred = np.where(miss, 0, np.where(false_alarm, 1, y))
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    fn = result.false_negative_patterns()
    assert fn and fn[0].conditions[0].feature == "transactions"
    assert fn[0].metric_name == "false negative rate"
    assert fn[0].baseline == pytest.approx(result.performance.false_negative_rate)
    assert not any(c.feature == "transactions" and c.op == ">" for p in
                   result.false_positive_patterns() for c in p.conditions)


def test_multiclass(multiclass_result):
    r = multiclass_result
    assert not r.performance.is_binary
    assert {str(c.label) for c in r.performance.per_class} == {"bird", "cat", "dog"}
    assert r.confusion_matrix().shape == (3, 3)
    birds = r.class_patterns("bird")
    assert birds and str(birds[0].conditions[0]) == "region == north"
    assert birds[0].metric_name == "miss rate for class bird"
    assert r.performance.per_class[0].n_missed >= 0
    with pytest.raises(UnsupportedTaskError):
        r.false_positives()
    with pytest.raises(UnsupportedTaskError):
        r.residual_analysis()
    pairs = {(c.true_label, c.predicted_label) for c in r.error_summary.top_confusions[:2]}
    assert ("bird", "dog") in pairs


def test_imbalanced_classification():
    n = 10000
    rng = np.random.default_rng(31)
    X = make_features(n, 31)
    y = (rng.random(n) < 0.02).astype(int)
    y_pred = np.where(rng.random(n) < 0.03, 1 - y, y)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert result.performance.is_binary
    assert result.dataset.target_distribution["1"] < 400
    assert isinstance(result.summary(print_summary=False), str)


def test_balanced_classification_with_real_model():
    n = 6000
    rng = np.random.default_rng(41)
    X = make_features(n, 41)
    score = 0.00005 * (X["income"] - 40000) + 0.15 * X["account_age"] - 1.5
    y = (score + rng.normal(0, 0.6, n) > 0).astype(int).to_numpy()
    hard = ((X["age"] < 25) & (X["income"] < 30000)).to_numpy()
    y[hard] = 1 - y[hard]  # concept differs in a subgroup absent from the training data
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["region", "student"]),
                                remainder="passthrough"),
        HistGradientBoostingClassifier(max_iter=60, random_state=0))
    train = np.arange(3000)[~hard[:3000]]
    model.fit(X.iloc[train], y[train])
    result = ErrorLens(model, X.iloc[3000:], y[3000:]).analyze()
    assert result.performance.roc_auc is not None
    assert result.error_summary.mean_confidence_errors is not None
    truth = hard[3000:]
    X_test = X.iloc[3000:].reset_index(drop=True)
    assert any(jaccard(p.mask(X_test), truth) > 0.5 for p in result.patterns[:3])


def test_no_errors():
    X = make_features(500)
    y = np.random.default_rng(0).integers(0, 2, 500)
    result = ErrorLens(FixedPredictor(y), X, y).analyze()
    assert result.error_rate() == 0
    assert result.patterns == []
    assert result.feature_analysis == []
    assert any("no errors" in w for w in result.warnings)
    assert "No statistically significant" in result.summary(print_summary=False)


def test_extremely_high_error():
    X = make_features(500)
    y = np.random.default_rng(1).integers(0, 2, 500)
    result = ErrorLens(FixedPredictor(1 - y), X, y).analyze()
    assert result.error_rate() == 1.0
    assert result.patterns == []
    assert any("Every prediction is wrong" in w for w in result.warnings)


def test_string_labels_and_positive_label():
    X = make_features(2000)
    rng = np.random.default_rng(2)
    y = rng.choice(["churn", "stay"], 2000)
    y_pred = np.where(rng.random(2000) < 0.1, np.where(y == "churn", "stay", "churn"), y)
    result = ErrorLens.from_predictions(X, y, y_pred, positive_label="churn").analyze()
    assert result.performance.positive_label == "churn"
    with pytest.raises(InvalidInputError):
        ErrorLens.from_predictions(X, y, y_pred, positive_label="nope").analyze()


def test_label_mismatch_warning():
    X = make_features(300)
    y = np.array(["a", "b"] * 150)
    y_pred = np.array(["x", "y"] * 150)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    assert any("label encoding" in w for w in result.warnings)


def test_continuous_predictions_for_classification_rejected():
    X = make_features(300)
    y = np.array([0, 1] * 150)
    with pytest.raises(UnsupportedTaskError):
        ErrorLens.from_predictions(X, y, np.linspace(0, 1, 300),
                                   task="classification").analyze()


def test_frames_and_accessors(binary_result):
    pf = binary_result.patterns_frame()
    assert {"pattern", "lift", "p_value_adjusted", "kind"} <= set(pf.columns)
    assert set(pf["kind"]) <= {"error", "false_positive", "false_negative"}
    fa = binary_result.feature_associations()
    assert list(fa.columns[:3]) == ["feature", "type", "test"]
    assert isinstance(binary_result.confusion_matrix(), pd.DataFrame)
    assert "AnalysisResult(task='classification'" in repr(binary_result)
