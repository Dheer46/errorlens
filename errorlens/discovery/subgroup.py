"""Subgroup discovery engine: split → search → simplify → validate → correct → rank.

This module is where the statistical guarantees of ErrorLens are enforced. The essential
idea is *honest inference*: candidate patterns are generated on a discovery split and
tested on an independent validation split, so the hypotheses being tested are fixed before
the validation data are seen. The multiple-testing family is then exactly the set of tested
candidates, and the correction (Benjamini–Hochberg by default) is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from errorlens.analysis.statistics import (
    adjust_pvalues,
    binary_group_stats,
    continuous_group_stats,
    is_effectively_constant,
)
from errorlens.core.config import AnalysisConfig
from errorlens.core.types import (
    Condition,
    DiscoveryStatistics,
    FailurePattern,
    FeatureInfo,
    GroupStats,
)
from errorlens.discovery.deduplication import (
    deduplicate,
    is_meaningful_refinement,
    simplify,
)
from errorlens.discovery.items import build_items
from errorlens.discovery.ranking import COMPLEXITY_PENALTY, pattern_score, sort_key
from errorlens.discovery.search import beam_search, quality, refine_thresholds, tree_search

MIN_HOLDOUT_EVENTS = 10
HONEST_MIN_FACTOR = 5  # validation split must hold >= 5 * min_samples rows for "auto"


@dataclass
class DiscoveryTarget:
    """What to discover patterns for.

    Attributes:
        kind: Identifier such as ``"error"``, ``"false_negative"`` or ``"high_error"``.
        values: Per-row target: 0/1 events (binary) or a non-negative loss (continuous).
        population: Boolean mask of rows on which the target is defined.
        is_binary: Whether ``values`` is a binary event indicator.
        metric_name: Human-readable name of the subgroup metric (e.g. "error rate").
        target_label: Class label for class-specific targets.
        residuals: Signed residuals (regression) used to describe the bias of patterns.
    """

    kind: str
    values: np.ndarray
    population: np.ndarray
    is_binary: bool
    metric_name: str
    target_label: Any = None
    residuals: np.ndarray | None = None


@dataclass
class _Candidate:
    conditions: tuple[Condition, ...]
    disc_mask: np.ndarray
    disc_value: float
    disc_quality: float
    disc_lift: float


class SubgroupDiscovery:
    """Discovers statistically validated, de-duplicated failure patterns for one target."""

    def __init__(self, config: AnalysisConfig, feature_infos: list[FeatureInfo]) -> None:
        self.config = config
        self.feature_infos = feature_infos

    # ------------------------------------------------------------------------------
    def run(self, X: pd.DataFrame, target: DiscoveryTarget) -> tuple[list[FailurePattern],
                                                                     DiscoveryStatistics]:
        cfg = self.config
        pop = np.asarray(target.population, dtype=bool)
        Xp = X.loc[pop].reset_index(drop=True)
        t = np.asarray(target.values, dtype=float)[pop]
        residuals = None if target.residuals is None else np.asarray(target.residuals)[pop]
        N = len(t)
        n_events = int(t.sum()) if target.is_binary else None
        baseline = float(t.mean()) if N else float("nan")

        stats = DiscoveryStatistics(
            kind=target.kind, metric_name=target.metric_name, population_size=N,
            n_events=n_events, baseline=baseline, honest=False, n_discovery=0, n_validation=0,
            n_conditions=0, n_candidates_evaluated=0, n_candidates_tested=0, family_size=0,
            n_significant=0, n_reported=0, correction=cfg.correction,
            alpha=cfg.significance_level, target_label=target.target_label,
        )

        reason = self._skip_reason(N, t, target.is_binary)
        if reason:
            stats.note = reason
            return [], stats

        honest, note = self._decide_honest(N, n_events, target.is_binary)
        stats.honest = honest
        if note:
            stats.note = note
        rng = np.random.default_rng(cfg.random_state)
        idx_disc, idx_val = self._split(N, t, target.is_binary, honest, rng)
        if len(idx_disc) > cfg.max_discovery_rows:
            idx_disc = np.sort(rng.choice(idx_disc, cfg.max_discovery_rows, replace=False))
        stats.n_discovery, stats.n_validation = len(idx_disc), len(idx_val)

        Xd = Xp.iloc[idx_disc].reset_index(drop=True)
        td = t[idx_disc]
        min_support_disc = max(5, math.ceil(cfg.min_samples * len(idx_disc) / N))

        # 1. Generate candidates on the discovery split ---------------------------------
        items = build_items(Xd, self.feature_infos, n_bins=cfg.n_bins,
                            max_categories=cfg.max_categories, min_support=min_support_disc)
        stats.n_conditions = items.n_items
        beam = beam_search(items, td, min_support=min_support_disc, max_depth=cfg.max_depth,
                           beam_width=cfg.beam_width, pool_size=4 * cfg.max_candidates)
        raw = list(beam.candidates)
        n_evaluated = beam.n_evaluated
        if cfg.use_tree:
            tree = tree_search(Xd, self.feature_infos, td, is_binary=target.is_binary,
                               min_support=min_support_disc, max_depth=cfg.max_depth,
                               max_categories=cfg.max_categories,
                               random_state=cfg.random_state)
            raw += tree.candidates
            n_evaluated += tree.n_evaluated

        candidates = self._screen(raw, Xd, td, min_support_disc)
        stats.n_candidates_evaluated = max(n_evaluated, len(candidates))
        if not candidates:
            stats.note = stats.note or "No candidate subgroup exceeded the baseline."
            return [], stats

        # 2. Test candidates on the validation split -------------------------------------
        Xv = Xp.iloc[idx_val].reset_index(drop=True)
        tv = t[idx_val]
        inference = "holdout" if honest else "in-sample"
        val_stats = [self._group_stats(c.conditions, Xv, tv, target.is_binary, inference)
                     for c in candidates]
        pvalues = np.array([s.p_value if not math.isnan(s.p_value) else 1.0 for s in val_stats])
        family = len(candidates) if honest else stats.n_candidates_evaluated
        adjusted = adjust_pvalues(pvalues, cfg.correction, n_tests=family)
        stats.n_candidates_tested = len(candidates)
        stats.family_size = family

        # 3. Keep significant, practically relevant patterns ----------------------------
        patterns: list[FailurePattern] = []
        full_cache: dict[tuple[Condition, ...], np.ndarray] = {}
        for cand, vstat, padj in zip(candidates, val_stats, adjusted):
            significant = bool(padj < cfg.significance_level)
            if not significant or not (vstat.lift >= cfg.min_lift):
                continue
            full_mask = self._mask(cand.conditions, Xp, full_cache)
            if int(full_mask.sum()) < cfg.min_samples:
                continue
            if honest:
                fstat = self._group_stats(cand.conditions, Xp, t, target.is_binary,
                                          "descriptive", with_tests=False, mask=full_mask)
            else:
                fstat = vstat
            pattern = FailurePattern(
                conditions=cand.conditions, kind=target.kind, metric_name=target.metric_name,
                is_binary=target.is_binary, stats=fstat, validation=vstat,
                p_value_adjusted=float(padj), significant=significant,
                target_label=target.target_label, correction=cfg.correction,
                mean_residual=(float(residuals[full_mask].mean())
                               if residuals is not None else None),
            )
            pattern.score = pattern_score(pattern)
            patterns.append(pattern)
        stats.n_significant = len(patterns)

        # 4. Rank, de-duplicate on the full data, truncate --------------------------------
        patterns.sort(key=sort_key(cfg.rank_by))
        # Whether a sub-region is "detectably worse" than its parent is judged on the
        # validation split, which did not take part in selecting either pattern.
        val_cache: dict[tuple[Condition, ...], np.ndarray] = {}

        def refines(child: FailurePattern, parent: FailurePattern) -> bool:
            return is_meaningful_refinement(self._mask(child.conditions, Xv, val_cache),
                                            self._mask(parent.conditions, Xv, val_cache), tv)

        patterns = deduplicate(
            patterns,
            conditions_of=lambda p: p.conditions,
            mask_of=lambda p: self._mask(p.conditions, Xp, full_cache),
            target=t,
            min_support=cfg.min_samples,
            threshold=cfg.dedup_threshold,
            refines=refines,
        )[: cfg.max_patterns]
        for i, p in enumerate(patterns, start=1):
            p.rank = i
        stats.n_reported = len(patterns)
        return patterns, stats

    # ------------------------------------------------------------------------------
    def _skip_reason(self, N: int, t: np.ndarray, is_binary: bool) -> str:
        cfg = self.config
        if 2 * cfg.min_samples > N:
            return (f"Population too small ({N} rows) for min_samples={cfg.min_samples}; "
                    "pattern discovery skipped.")
        if is_binary:
            k = int(t.sum())
            if k == 0:
                return "No errors in this population; nothing to discover."
            if k == N:
                return "Every row in this population is an error; no subgroup can stand out."
        elif is_effectively_constant(t):
            return "The error metric is constant; no subgroup can stand out."
        if not any(f.kind != "skipped" for f in self.feature_infos):
            return "No usable features for pattern discovery."
        return ""

    def _decide_honest(self, N: int, n_events: int | None, is_binary: bool) -> tuple[bool, str]:
        cfg = self.config
        frac = cfg.holdout_fraction
        n_val = N * frac
        enough_rows = n_val >= HONEST_MIN_FACTOR * cfg.min_samples
        enough_events = True
        if is_binary and n_events is not None:
            enough_events = (n_events * frac >= MIN_HOLDOUT_EVENTS
                             and (N - n_events) * frac >= MIN_HOLDOUT_EVENTS)
        if cfg.honest == "auto":
            if enough_rows and enough_events:
                return True, ""
            return False, ("Too few rows/errors to hold out a validation split; patterns were "
                           "tested in-sample and are exploratory (p-values are optimistic).")
        if cfg.honest is True:
            if n_val >= 2 * cfg.min_samples and enough_events:
                return True, ""
            return False, ("honest=True requested but the population is too small to split; "
                           "fell back to exploratory in-sample testing.")
        return False, ("honest=False: patterns were tested on the same data used to find them; "
                       "p-values are optimistic and results are exploratory.")

    def _split(self, N: int, t: np.ndarray, is_binary: bool, honest: bool,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        all_idx = np.arange(N)
        if not honest:
            return all_idx, all_idx
        frac = self.config.holdout_fraction
        groups = [np.flatnonzero(t == 1), np.flatnonzero(t == 0)] if is_binary else [all_idx]
        val_parts = []
        for g in groups:
            perm = rng.permutation(g)
            val_parts.append(perm[: int(round(len(g) * frac))])
        val = np.sort(np.concatenate(val_parts))
        disc = np.setdiff1d(all_idx, val, assume_unique=True)
        return disc, val

    def _screen(self, raw: list[tuple[Condition, ...]], Xd: pd.DataFrame, td: np.ndarray,
                min_support: int) -> list[_Candidate]:
        """Simplify, filter and de-duplicate candidates on the discovery split."""
        cfg = self.config
        cache: dict[tuple[Condition, ...], np.ndarray] = {}
        cond_cache: dict[Condition, np.ndarray] = {}

        def mask_of(conds: tuple[Condition, ...]) -> np.ndarray:
            key = tuple(sorted(conds, key=str))
            if key not in cache:
                m = np.ones(len(Xd), dtype=bool)
                for c in key:
                    if c not in cond_cache:
                        cond_cache[c] = c.mask(Xd)
                    m &= cond_cache[c]
                cache[key] = m
            return cache[key]

        mu = float(td.mean())
        sd = float(td.std())
        seen: set[tuple[str, ...]] = set()
        out: list[_Candidate] = []
        for conds in raw:
            conds = simplify(conds, mask_of, td, min_support)
            key = tuple(sorted(str(c) for c in conds))
            if key in seen:
                continue
            seen.add(key)
            conds = tuple(sorted(conds, key=lambda c: c.feature))
            m = mask_of(conds)
            n = int(m.sum())
            if n < min_support or n == len(td):
                continue
            value = float(td[m].mean())
            lift = value / mu if mu > 0 else float("nan")
            if not lift >= cfg.min_lift:
                continue
            q = float(quality(np.array([n]), np.array([value]), mu, sd)[0])
            q /= 1.0 + COMPLEXITY_PENALTY * (len(conds) - 1)
            out.append(_Candidate(conds, m, value, q, lift))
        out.sort(key=lambda c: -c.disc_quality)
        out = deduplicate(out, conditions_of=lambda c: c.conditions, mask_of=lambda c: c.disc_mask,
                          target=td, min_support=min_support, threshold=cfg.dedup_threshold)
        out = out[: cfg.max_candidates]

        # Sharpen numeric thresholds of the shortlist on a finer grid, then re-screen.
        refined: list[_Candidate] = []
        seen_refined: set[tuple[str, ...]] = set()
        for cand in out:
            conds = refine_thresholds(cand.conditions, Xd, td, min_support=min_support,
                                      mu=mu, sd=sd)
            key = tuple(sorted(str(c) for c in conds))
            if key in seen_refined:
                continue
            seen_refined.add(key)
            m = mask_of(conds)
            n = int(m.sum())
            value = float(td[m].mean()) if n else 0.0
            q = float(quality(np.array([n]), np.array([value]), mu, sd)[0])
            q /= 1.0 + COMPLEXITY_PENALTY * (len(conds) - 1)
            refined.append(_Candidate(conds, m, value, q, value / mu if mu > 0 else 0.0))
        refined.sort(key=lambda c: -c.disc_quality)
        return deduplicate(refined, conditions_of=lambda c: c.conditions,
                           mask_of=lambda c: c.disc_mask, target=td, min_support=min_support,
                           threshold=cfg.dedup_threshold)

    @staticmethod
    def _mask(conds: tuple[Condition, ...], X: pd.DataFrame,
              cache: dict[tuple[Condition, ...], np.ndarray]) -> np.ndarray:
        if conds not in cache:
            m = np.ones(len(X), dtype=bool)
            for c in conds:
                m &= c.mask(X)
            cache[conds] = m
        return cache[conds]

    @staticmethod
    def _group_stats(conds: tuple[Condition, ...], X: pd.DataFrame, t: np.ndarray,
                     is_binary: bool, inference: str, with_tests: bool = True,
                     mask: np.ndarray | None = None) -> GroupStats:
        if mask is None:
            mask = np.ones(len(X), dtype=bool)
            for c in conds:
                mask &= c.mask(X)
        if is_binary:
            return binary_group_stats(mask, t, inference=inference, with_tests=with_tests)
        return continuous_group_stats(mask, t, inference=inference, with_tests=with_tests)
