"""Regression performance metrics and residual diagnostics (residual = y_true - y_pred)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn import metrics as skm

from errorlens.analysis.statistics import adjust_pvalues, is_effectively_constant
from errorlens.core.types import ErrorSummary, FeatureInfo, RegressionPerformance, ResidualAnalysis


def regression_performance(y: np.ndarray, y_pred: np.ndarray) -> RegressionPerformance:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y - y_pred
    mse = float(skm.mean_squared_error(y, y_pred))
    perf = RegressionPerformance(
        n_samples=len(y),
        mae=float(skm.mean_absolute_error(y, y_pred)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        median_ae=float(skm.median_absolute_error(y, y_pred)),
        r2=float(skm.r2_score(y, y_pred)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
        max_error=float(np.max(np.abs(residual))),
        mean_residual=float(residual.mean()),
    )
    if relative_error_appropriate(y):
        perf.mape = float(np.mean(np.abs(residual) / np.abs(y)))
    return perf


def relative_error_appropriate(y: np.ndarray) -> bool:
    """Relative errors are only meaningful when the target is bounded away from zero."""
    ay = np.abs(np.asarray(y, dtype=float))
    if len(ay) == 0:
        return False
    return bool(ay.min() > 1e-8 * max(float(ay.max()), 1.0) and (ay > 0).all())


def regression_error_summary(y: np.ndarray, y_pred: np.ndarray, tau: float) -> ErrorSummary:
    residual = np.asarray(y, dtype=float) - np.asarray(y_pred, dtype=float)
    severe = np.abs(residual) > tau
    return ErrorSummary(
        n_errors=int(severe.sum()),
        error_rate=float(severe.mean()),
        n_underpredicted=int((residual > 0).sum()),
        n_overpredicted=int((residual < 0).sum()),
        severe_threshold=tau,
        n_severe_under=int((residual > tau).sum()),
        n_severe_over=int((residual < -tau).sum()),
    )


def residual_analysis(X: pd.DataFrame, y: np.ndarray, y_pred: np.ndarray,
                      infos: list[FeatureInfo], tau: float) -> ResidualAnalysis:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r = y - y_pred
    abs_r = np.abs(r)
    under, over = r > 0, r < 0
    n = len(r)

    if n > 2 and not is_effectively_constant(r):
        bias_p = float(sps.ttest_1samp(r, 0.0).pvalue)
        skew = float(sps.skew(r))
        kurt = float(sps.kurtosis(r))
    else:
        bias_p, skew, kurt = 1.0, 0.0, 0.0
    if n > 2 and not is_effectively_constant(y_pred) and not is_effectively_constant(abs_r):
        rho_res = sps.spearmanr(y_pred, abs_r)
        hetero_rho, hetero_p = float(rho_res.statistic), float(rho_res.pvalue)
    else:
        hetero_rho, hetero_p = 0.0, 1.0

    rel = None
    if relative_error_appropriate(y):
        rel = float(np.median(abs_r / np.abs(y)))

    return ResidualAnalysis(
        mean_residual=float(r.mean()),
        median_residual=float(np.median(r)),
        std_residual=float(r.std(ddof=1)) if n > 1 else 0.0,
        skewness=skew,
        excess_kurtosis=kurt,
        pct_underpredicted=float(under.mean()),
        pct_overpredicted=float(over.mean()),
        mean_underprediction=float(r[under].mean()) if under.any() else 0.0,
        mean_overprediction=float(-r[over].mean()) if over.any() else 0.0,
        severe_threshold=tau,
        n_severe_under=int((r > tau).sum()),
        n_severe_over=int((r < -tau).sum()),
        heteroscedasticity_rho=hetero_rho,
        heteroscedasticity_p=hetero_p,
        bias_test_p=bias_p,
        median_relative_error=rel,
        residual_by_prediction_decile=_by_prediction_bins(y, y_pred),
        feature_residual_correlations=_feature_residual_correlations(X, r, infos),
    )


def _by_prediction_bins(y: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    if len(np.unique(y_pred)) < 2:
        return []
    codes = pd.qcut(y_pred, q=min(bins, len(np.unique(y_pred))), labels=False, duplicates="drop")
    out = []
    for b in np.unique(codes):
        m = codes == b
        out.append({
            "bin": int(b) + 1,
            "n": int(m.sum()),
            "pred_min": float(y_pred[m].min()),
            "pred_max": float(y_pred[m].max()),
            "mean_predicted": float(y_pred[m].mean()),
            "mean_actual": float(y[m].mean()),
            "mean_residual": float((y[m] - y_pred[m]).mean()),
            "mae": float(np.abs(y[m] - y_pred[m]).mean()),
        })
    return out


def _feature_residual_correlations(X: pd.DataFrame, r: np.ndarray,
                                   infos: list[FeatureInfo]) -> list[dict[str, Any]]:
    """Association between each feature and the *signed* residual (systematic bias)."""
    rows: list[dict[str, Any]] = []
    for info in infos:
        if info.kind == "skipped":
            continue
        col = X[info.name]
        present = col.notna().to_numpy()
        if present.sum() < 10:
            continue
        if info.kind == "numeric":
            x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            ok = ~np.isnan(x)
            if np.std(x[ok]) == 0:
                continue
            res = sps.spearmanr(x[ok], r[ok])
            rows.append({"feature": info.name, "measure": "spearman_rho",
                         "value": float(res.statistic), "p_value": float(res.pvalue)})
        else:
            groups = [r[(col == lv).to_numpy(dtype=bool, na_value=False)]
                      for lv in col.value_counts().index[:30]]
            groups = [g for g in groups if len(g) >= 5]
            if len(groups) < 2:
                continue
            h, p = sps.kruskal(*groups)
            n_used = sum(len(g) for g in groups)
            eps2 = float(h) / (n_used - 1) if n_used > 1 else float("nan")
            rows.append({"feature": info.name, "measure": "epsilon_squared",
                         "value": eps2, "p_value": float(p)})
    if rows:
        adj = adjust_pvalues(np.array([row["p_value"] for row in rows]), "fdr_bh")
        for row, a in zip(rows, adj):
            row["p_value_adjusted"] = float(a)
    rows.sort(key=lambda row: -abs(row["value"]) if row["value"] == row["value"] else 0)
    return rows
