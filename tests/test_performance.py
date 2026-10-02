import time

import numpy as np
import pytest

from errorlens import ErrorLens
from tests.conftest import make_features


@pytest.mark.slow
def test_tens_of_thousands_of_rows_is_fast():
    n = 50_000
    rng = np.random.default_rng(0)
    X = make_features(n)
    for i in range(14):  # 20 features in total
        X[f"noise_{i}"] = rng.normal(size=n)
    y = rng.integers(0, 2, n)
    hard = ((X["age"] < 25) & (X["income"] < 30000)).to_numpy()
    y_pred = np.where(rng.random(n) < np.where(hard, 0.4, 0.05), 1 - y, y)
    start = time.perf_counter()
    result = ErrorLens.from_predictions(X, y, y_pred).analyze()
    elapsed = time.perf_counter() - start
    assert result.patterns
    assert elapsed < 60, f"analysis took {elapsed:.1f}s"
