"""JSON export — a direct serialization of :meth:`AnalysisResult.to_dict`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult


def write_json(result: AnalysisResult, path: str | Path | None = None, indent: int = 2) -> str:
    """Serialize ``result`` to strict JSON (no NaN/Infinity); write to ``path`` if given."""
    text = json.dumps(result.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text
