import math

import numpy as np
import pytest
from scipy import stats as sps

from errorlens.analysis.statistics import (
    adjust_pvalues,
    binary_group_stats,
    cohens_d,
    cohens_h,
    continuous_group_stats,
    fisher_greater,
    magnitude_label,
    mean_interval,
    relative_risk_interval,
    welch_greater,
    wilson_interval,
)


class TestAdjustPvalues:
    p = np.array([0.01, 0.04, 0.03, 0.005])

    def test_benjamini_hochberg_known_values(self):
        np.testing.assert_allclose(adjust_pvalues(self.p, "fdr_bh"), [0.02, 0.04, 0.04, 0.02])

    def test_bonferroni(self):
        np.testing.assert_allclose(adjust_pvalues(self.p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])

    def test_holm(self):
        # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .02,.03,.06,.06
        np.testing.assert_allclose(adjust_pvalues(self.p, "holm"), [0.03, 0.06, 0.06, 0.02])

    def test_by_is_more_conservative_than_bh(self):
        assert np.all(adjust_pvalues(self.p, "fdr_by") >= adjust_pvalues(self.p, "fdr_bh"))

    def test_none(self):
        np.testing.assert_allclose(adjust_pvalues(self.p, "none"), self.p)

    def test_larger_family_is_more_conservative(self):
        small = adjust_pvalues(self.p, "fdr_bh")
        large = adjust_pvalues(self.p, "fdr_bh", n_tests=100)
        assert np.all(large >= small)
        # BH with 96 extra p=1 hypotheses equals BH computed on the padded vector
        padded = np.concatenate([self.p, np.ones(96)])
        np.testing.assert_allclose(large, adjust_pvalues(padded, "fdr_bh")[:4])

    def test_capped_at_one_and_empty(self):
        assert adjust_pvalues(np.array([0.9, 0.8]), "bonferroni").max() <= 1.0
        assert len(adjust_pvalues(np.array([]), "fdr_bh")) == 0

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            adjust_pvalues(self.p, "magic")


def test_fisher_greater_matches_scipy():
    k1, n1, k0, n0 = 30, 100, 50, 900
    expected = sps.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater").pvalue
    assert fisher_greater(k1, n1, k1 + k0, n1 + n0) == pytest.approx(expected, rel=1e-9)


def test_fisher_greater_degenerate():
    assert fisher_greater(0, 0, 5, 100) == 1.0
    assert fisher_greater(5, 100, 5, 100) == 1.0


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_mean_interval_contains_mean():
    x = np.random.default_rng(0).normal(5, 1, 200)
    lo, hi = mean_interval(x)
    assert lo < x.mean() < hi


def test_relative_risk_handles_zero_cells():
    rr, lo, hi = relative_risk_interval(10, 50, 0, 100)
    assert math.isfinite(rr) and rr > 1
    assert lo < rr < hi


def test_effect_sizes_sign():
    assert cohens_h(0.5, 0.1) > 0 > cohens_h(0.1, 0.5)
    a, b = np.array([3.0, 4, 5, 6]), np.array([1.0, 2, 3, 2])
    assert cohens_d(a, b) > 0
    assert magnitude_label(0.85, "cohens_h") == "large"
    assert magnitude_label(0.05, "cramers_v") == "negligible"


def test_welch_greater_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 1, 50), rng.normal(0, 2, 80)
    expected = sps.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue
    assert welch_greater(a, b) == pytest.approx(expected)


def test_binary_group_stats():
    events = np.array([1] * 20 + [0] * 80 + [1] * 10 + [0] * 890, dtype=float)
    mask = np.zeros(1000, dtype=bool)
    mask[:100] = True
    s = binary_group_stats(mask, events)
    assert s.n == 100 and s.n_events == 20
    assert s.value == pytest.approx(0.2)
    assert s.baseline == pytest.approx(0.03)
    assert s.lift == pytest.approx(0.2 / 0.03)
    assert s.relative_risk == pytest.approx(0.2 / (10 / 900))
    assert s.event_coverage == pytest.approx(20 / 30)
    assert s.ci_low < 0.2 < s.ci_high
    assert s.p_value < 1e-6


def test_continuous_group_stats():
    rng = np.random.default_rng(0)
    values = np.abs(rng.normal(0, 1, 1000))
    mask = np.zeros(1000, dtype=bool)
    mask[:100] = True
    values[:100] *= 3
    s = continuous_group_stats(mask, values)
    assert s.lift > 2
    assert s.p_value < 1e-6
    assert s.effect_size > 0.8
    assert s.n_events is None
