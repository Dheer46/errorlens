"""Helpers for optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from errorlens.exceptions import MissingDependencyError


def require(module: str, *, package: str, extra: str, feature: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` with install instructions."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatching in tests
        raise MissingDependencyError(package=package, extra=extra, feature=feature) from exc


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
