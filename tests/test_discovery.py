"""Pattern discovery on synthetic data where the true failure pattern is known."""

import numpy as np
import pandas as pd
import pytest

from errorlens import Condition, ErrorLens
from errorlens.core.config import AnalysisConfig
from errorlens.core.data import infer_feature_types
from errorlens.discovery.deduplication import jaccard
from errorlens.discovery.items import build_items, nice_threshold
from errorlens.discovery.subgroup import DiscoveryTarget, SubgroupDiscovery
from tests.conftest import make_binary_case, make_features, planted_subgroup


def _discover(X, errors, **cfg):
    config = AnalysisConfig(**cfg)
    infos = infer_feature_types(X, config)
    target = DiscoveryTarget("error", errors.astype(float), np.ones(len(X), bool), True,
                             "error rate")
    return SubgroupDiscovery(config, infos).run(X, target)


def test_planted_pattern_is_top_ranked(binary_case, binary_result):
    X, _, _ = binary_case
    top = binary_result.patterns[0]
    assert {c.feature for c in top.conditions} == {"age", "income"}
    truth = planted_subgroup(X)
    assert jaccard(top.mask(X), truth) > 0.75
    assert top.lift > 3
    assert top.p_value_adjusted < 1e-6
    assert top.validation.inference == "holdout"


def test_missing_value_pattern_is_found():
    rng = np.random.default_rng(3)
    X = make_features(8000, 3)
    errors = rng.random(8000) < np.where(X["transactions"].isna(), 0.35, 0.05)
    patterns, _ = _discover(X, errors)
    assert any(c.op == "is_missing" and c.feature == "transactions"
               for p in patterns[:2] for c in p.conditions)


def test_categorical_interaction_pattern_is_found():
    rng = np.random.default_rng(4)
    X = make_features(10000, 4)
    hard = ((X["region"] == "east") & X["student"]).to_numpy()
    errors = rng.random(10000) < np.where(hard, 0.45, 0.06)
    patterns, _ = _discover(X, errors)
    top = patterns[0]
    assert {str(c) for c in top.conditions} == {"region == east", "student == True"}


def test_null_data_yields_no_patterns():
    """Multiple-testing control: pure noise should (almost) never produce findings."""
    X = make_features(6000, 7)
    total = 0
    for seed in range(5):
        errors = np.random.default_rng(100 + seed).random(6000) < 0.1
        patterns, stats = _discover(X, errors)
        total += len(patterns)
        assert stats.n_candidates_evaluated > 100  # many hypotheses were considered
    assert total <= 1


def test_min_samples_is_respected():
    X, y, y_pred = make_binary_case(n=6000, seed=2)
    result = ErrorLens.from_predictions(X, y, y_pred, min_samples=80).analyze()
    assert result.patterns
    for p in result.all_patterns():
        assert p.n_samples >= 80


def test_tiny_planted_group_below_min_samples_not_reported():
    rng = np.random.default_rng(11)
    X = make_features(4000, 11)
    hard = np.zeros(4000, dtype=bool)
    hard[:20] = True
    X.loc[hard, "account_age"] = 99.0
    errors = rng.random(4000) < np.where(hard, 0.9, 0.05)
    patterns, _ = _discover(X, errors, min_samples=30)
    for p in patterns:
        assert p.n_samples >= 30


def test_patterns_are_significant_and_deduplicated(binary_case, binary_result):
    X, _, _ = binary_case
    cfg = binary_result.metadata.config
    for p in binary_result.patterns:
        assert p.significant
        assert p.p_value_adjusted < cfg["significance_level"]
        assert p.validation.lift >= cfg["min_lift"]
    masks = [p.mask(X) for p in binary_result.patterns]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert jaccard(masks[i], masks[j]) < cfg["dedup_threshold"]


def test_no_near_duplicate_refinements(binary_result):
    descs = [set(map(str, p.conditions)) for p in binary_result.patterns]
    for i, a in enumerate(descs):
        for j, b in enumerate(descs):
            if i != j and a < b:
                # a refinement survived: it must be materially worse than its parent
                assert binary_result.patterns[j].rate >= binary_result.patterns[i].rate * 1.1


def test_ranking_by_score_and_lift():
    X, y, y_pred = make_binary_case(n=8000, seed=4)
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    scores = [p.score for p in result.patterns]
    assert scores == sorted(scores, reverse=True)
    assert [p.rank for p in result.patterns] == list(range(1, len(result.patterns) + 1))
    by_lift = ErrorLens.from_predictions(X, y, y_pred, rank_by="lift").analyze()
    lifts = [p.lift for p in by_lift.patterns]
    assert lifts == sorted(lifts, reverse=True)


def test_max_patterns_is_respected():
    X, y, y_pred = make_binary_case(n=8000, seed=6)
    result = ErrorLens.from_predictions(X, y, y_pred, max_patterns=1).analyze()
    assert len(result.patterns) == 1
    assert all(len(v) <= 1 for v in result.directional_patterns.values())


def test_correction_methods_are_recorded_and_ordered():
    X, y, y_pred = make_binary_case(n=3000, seed=8, subgroup_error=0.25)
    counts = {}
    for method in ("none", "fdr_bh", "bonferroni"):
        result = ErrorLens.from_predictions(X, y, y_pred, correction=method,
                                            analyze_directional=False).analyze()
        st = result.discovery_statistics[0]
        assert st.correction == method
        assert st.family_size >= st.n_candidates_tested
        counts[method] = st.n_significant
    assert counts["bonferroni"] <= counts["fdr_bh"] <= counts["none"]


def test_in_sample_mode_is_flagged():
    X, y, y_pred = make_binary_case(n=3000, seed=9)
    result = ErrorLens.from_predictions(X, y, y_pred, honest=False).analyze()
    st = result.discovery_statistics[0]
    assert not st.honest
    assert st.family_size == st.n_candidates_evaluated  # whole search counted
    assert any("exploratory" in w for w in result.warnings)
    assert all(p.validation.inference == "in-sample" for p in result.patterns)


def test_small_data_falls_back_to_in_sample():
    X, y, y_pred = make_binary_case(n=250, seed=10, subgroup_error=0.6)
    result = ErrorLens.from_predictions(X, y, y_pred, min_samples=20).analyze()
    assert not result.discovery_statistics[0].honest


def test_condition_mask_semantics():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "c": ["x", None, "y", "x"]})
    assert Condition("a", "<=", 2).mask(X).tolist() == [True, True, False, False]
    assert Condition("a", ">", 2).mask(X).tolist() == [False, False, False, True]
    assert Condition("a", "is_missing").mask(X).tolist() == [False, False, True, False]
    assert Condition("a", "in_range", 1, 4).mask(X).tolist() == [False, True, False, True]
    assert Condition("c", "==", "x").mask(X).tolist() == [True, False, False, True]
    assert Condition("c", "!=", "x").mask(X).tolist() == [False, False, True, False]
    assert str(Condition("a", "in_range", 1, 4)) == "1 < a <= 4"
    with pytest.raises(ValueError):
        Condition("a", "~", 1)


def test_pattern_mask_applies_to_new_data(binary_result):
    new = make_features(500, seed=99)
    mask = binary_result.patterns[0].mask(new)
    assert mask.dtype == bool and len(mask) == 500


def test_items_and_thresholds():
    assert nice_threshold(30193.4, integer_valued=True) == 30200
    assert nice_threshold(0.82345, integer_valued=False) == 0.823
    X = make_features(1000)
    config = AnalysisConfig()
    items = build_items(X, infer_feature_types(X, config), n_bins=10, max_categories=30,
                        min_support=15)
    assert items.matrix.shape == (1000, items.n_items)
    texts = [str(c) for c in items.conditions]
    assert "transactions is missing" in texts
    assert "region == north" in texts
    # every item's mask equals the condition evaluated on the raw data
    for j, cond in enumerate(items.conditions):
        assert np.array_equal(items.matrix[:, j], cond.mask(X))
