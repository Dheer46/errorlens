"""Exception hierarchy for ErrorLens.

Every error raised deliberately by ErrorLens derives from :class:`ErrorLensError`, so callers
can catch library errors with a single ``except`` clause. Several classes also derive from a
built-in exception (``ValueError``, ``ImportError``) so idiomatic handlers keep working.
"""

from __future__ import annotations


class ErrorLensError(Exception):
    """Base class for all ErrorLens errors."""


class InvalidInputError(ErrorLensError, ValueError):
    """Inputs (X, y, predictions, configuration) are malformed or inconsistent."""


class InsufficientDataError(InvalidInputError):
    """There are too few samples to perform the requested analysis."""


class UnsupportedTaskError(InvalidInputError):
    """The requested or inferred task is not supported."""


class ModelError(ErrorLensError, TypeError):
    """The model object does not expose the interface ErrorLens needs."""


class PredictionError(ErrorLensError, RuntimeError):
    """The model raised an exception (or returned invalid output) while predicting."""


class MissingDependencyError(ErrorLensError, ImportError):
    """An optional dependency required for the requested feature is not installed."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        self.package = package
        self.extra = extra
        self.feature = feature
        super().__init__(
            f"{feature} requires the optional dependency '{package}', which is not installed.\n"
            f"Install it with:\n\n    pip install \"errorlens[{extra}]\"\n\n"
            f"or directly:\n\n    pip install {package}"
        )
