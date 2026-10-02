"""Number formatting and console helpers shared by summaries and reports."""

from __future__ import annotations

import math
import sys
from typing import Any


def is_missing_number(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fmt_pct(value: float | None, digits: int = 1) -> str:
    """Format a proportion (0-1) as a percentage."""
    if is_missing_number(value):
        return "n/a"
    assert value is not None
    return f"{100.0 * value:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    """Format a number with ``digits`` significant digits, using thousands separators."""
    if is_missing_number(value):
        return "n/a"
    assert value is not None
    if isinstance(value, (bool,)):
        return str(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    is_whole = isinstance(value, int) or (float(value).is_integer() and abs(value) >= 1)
    if is_whole and abs(value) < 1e15:
        return f"{int(round(value)):,}"
    av = abs(value)
    if av == 0:
        return "0"
    if av >= 1e6 or av < 1e-3:
        return f"{value:.{max(digits - 1, 0)}e}"
    magnitude = int(math.floor(math.log10(av)))
    decimals = max(digits - 1 - magnitude, 0)
    return f"{value:,.{decimals}f}"


def fmt_int(value: int | float | None) -> str:
    if is_missing_number(value):
        return "n/a"
    assert value is not None
    return f"{int(value):,}"


def fmt_lift(value: float | None) -> str:
    if is_missing_number(value):
        return "n/a"
    assert value is not None
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}x"


def fmt_p(value: float | None) -> str:
    """Format a p-value the way reports conventionally do."""
    if is_missing_number(value):
        return "n/a"
    assert value is not None
    if value < 0.001:
        return "< 0.001"
    if value < 0.1:
        # Two significant digits, truncated so a value is never rounded up across a
        # conventional threshold (0.0497 -> "0.049", not "0.05").
        digits = 1 - math.floor(math.log10(value))
        factor = 10**digits
        return f"{math.floor(value * factor + 1e-9) / factor:.{digits}f}"
    return f"{value:.2f}" if value < 0.995 else "1.00"


def safe_print(text: str) -> None:
    """Print text, degrading gracefully on consoles that cannot encode Unicode."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        replacements = {"─": "-", "═": "=", "×": "x", "≥": ">=", "≤": "<=", "•": "*", "≠": "!=",
                        "—": "-", "–": "-", "²": "2", "≈": "~"}
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
