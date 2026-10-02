"""Analysis configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from errorlens.exceptions import InvalidInputError

Task = Literal["auto", "classification", "regression"]
Correction = Literal["fdr_bh", "fdr_by", "holm", "bonferroni", "none"]
RankBy = Literal["score", "lift", "p_value", "coverage", "error_coverage"]

VALID_TASKS = ("auto", "classification", "regression")
VALID_CORRECTIONS = ("fdr_bh", "fdr_by", "holm", "bonferroni", "none")
VALID_RANK_BY = ("score", "lift", "p_value", "coverage", "error_coverage")


@dataclass
class AnalysisConfig:
    """All tunable parameters of an ErrorLens analysis.

    The defaults are chosen to be statistically conservative and fast on tens of thousands
    of rows. Every parameter can also be passed directly to :class:`errorlens.ErrorLens`.

    Attributes:
        task: ``"auto"``, ``"classification"`` or ``"regression"``.
        max_patterns: Maximum number of failure patterns reported per discovery target.
        min_samples: Minimum number of rows a reported pattern must contain (full data).
        significance_level: Level ``alpha`` for the (corrected) hypothesis tests.
        correction: Multiple-testing correction: ``"fdr_bh"`` (Benjamini–Hochberg, default),
            ``"fdr_by"`` (Benjamini–Yekutieli), ``"holm"``, ``"bonferroni"`` or ``"none"``.
        max_depth: Maximum number of conditions in a pattern.
        beam_width: Number of subgroups kept per level of the beam search.
        n_bins: Number of quantile bins used to create numeric thresholds.
        max_categories: Maximum number of levels per categorical feature turned into conditions.
        honest: Sample-splitting policy. ``"auto"`` splits when data are large enough,
            ``True`` always splits, ``False`` never splits (exploratory in-sample inference).
        holdout_fraction: Fraction of rows reserved for validation when splitting.
        min_lift: Minimum validated lift for a pattern to be reported.
        dedup_threshold: Jaccard similarity above which two patterns are considered duplicates.
        rank_by: Ranking criterion for reported patterns.
        max_candidates: Maximum number of candidates carried from discovery into testing.
        use_tree: Also use a surrogate error tree as a candidate generator.
        features: Restrict discovery to these features (default: all usable features).
        ignore_features: Features excluded from discovery and feature analysis.
        positive_label: Positive class for binary classification (default: the larger label).
        max_discovery_rows: Subsample the discovery split above this many rows (speed).
        analyze_directional: Discover FP/FN (classification) or under/over-prediction
            (regression) patterns in addition to overall error patterns.
        max_classes: Maximum number of classes analysed individually (multiclass).
        random_state: Seed for the discovery/validation split and subsampling.
    """

    task: Task = "auto"
    max_patterns: int = 20
    min_samples: int = 30
    significance_level: float = 0.05
    correction: Correction = "fdr_bh"
    max_depth: int = 3
    beam_width: int = 25
    n_bins: int = 10
    max_categories: int = 30
    honest: bool | Literal["auto"] = "auto"
    holdout_fraction: float = 0.5
    min_lift: float = 1.1
    dedup_threshold: float = 0.7
    rank_by: RankBy = "score"
    max_candidates: int = 60
    use_tree: bool = True
    features: list[str] | None = None
    ignore_features: list[str] = field(default_factory=list)
    positive_label: Any = None
    max_discovery_rows: int = 50_000
    analyze_directional: bool = True
    max_classes: int = 10
    random_state: int | None = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`InvalidInputError` if any parameter is out of range."""
        if self.task not in VALID_TASKS:
            raise InvalidInputError(f"task must be one of {VALID_TASKS}, got {self.task!r}")
        if self.correction not in VALID_CORRECTIONS:
            raise InvalidInputError(
                f"correction must be one of {VALID_CORRECTIONS}, got {self.correction!r}"
            )
        if self.rank_by not in VALID_RANK_BY:
            raise InvalidInputError(f"rank_by must be one of {VALID_RANK_BY}, got {self.rank_by!r}")
        _check_int(self.max_patterns, "max_patterns", minimum=1)
        _check_int(self.min_samples, "min_samples", minimum=2)
        _check_int(self.max_depth, "max_depth", minimum=1)
        _check_int(self.beam_width, "beam_width", minimum=1)
        _check_int(self.n_bins, "n_bins", minimum=2)
        _check_int(self.max_categories, "max_categories", minimum=1)
        _check_int(self.max_candidates, "max_candidates", minimum=1)
        _check_int(self.max_discovery_rows, "max_discovery_rows", minimum=100)
        _check_int(self.max_classes, "max_classes", minimum=1)
        if not 0.0 < float(self.significance_level) < 1.0:
            raise InvalidInputError(
                f"significance_level must be in (0, 1), got {self.significance_level!r}"
            )
        if not 0.1 <= float(self.holdout_fraction) <= 0.9:
            raise InvalidInputError(
                f"holdout_fraction must be in [0.1, 0.9], got {self.holdout_fraction!r}"
            )
        if not 0.0 < float(self.dedup_threshold) <= 1.0:
            raise InvalidInputError(
                f"dedup_threshold must be in (0, 1], got {self.dedup_threshold!r}"
            )
        if float(self.min_lift) < 1.0:
            raise InvalidInputError(f"min_lift must be >= 1.0, got {self.min_lift!r}")
        if self.honest not in (True, False, "auto"):
            raise InvalidInputError(f"honest must be True, False or 'auto', got {self.honest!r}")
        if self.features is not None:
            self.features = [str(f) for f in self.features]
        self.ignore_features = [str(f) for f in self.ignore_features]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["positive_label"] is not None:
            data["positive_label"] = _jsonable(data["positive_label"])
        return data


def _check_int(value: Any, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidInputError(f"{name} must be >= {minimum}, got {value}")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
