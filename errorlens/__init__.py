"""ErrorLens — automatic, statistically rigorous failure-pattern discovery for ML models.

Quick start::

    from errorlens import ErrorLens

    result = ErrorLens(model, X_test, y_test).analyze()
    result.summary()
    result.export("report.html")
"""

from errorlens._version import __version__
from errorlens.core.analyzer import ErrorLens
from errorlens.core.config import AnalysisConfig
from errorlens.core.results import AnalysisResult
from errorlens.core.types import Condition, FailurePattern, GroupStats
from errorlens.exceptions import (
    ErrorLensError,
    InsufficientDataError,
    InvalidInputError,
    MissingDependencyError,
    ModelError,
    PredictionError,
    UnsupportedTaskError,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "Condition",
    "ErrorLens",
    "ErrorLensError",
    "FailurePattern",
    "GroupStats",
    "InsufficientDataError",
    "InvalidInputError",
    "MissingDependencyError",
    "ModelError",
    "PredictionError",
    "UnsupportedTaskError",
    "__version__",
]
