"""Export dispatch: infer the output format from the file extension."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from errorlens.exceptions import InvalidInputError

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult

FORMATS = {
    ".html": "html", ".htm": "html", ".pdf": "pdf", ".md": "markdown",
    ".markdown": "markdown", ".json": "json",
}
ALIASES = {"html": "html", "pdf": "pdf", "md": "markdown", "markdown": "markdown",
           "json": "json"}


def infer_format(path: str | Path, format: str | None = None) -> str:
    if format is not None:
        key = format.lower().lstrip(".")
        if key not in ALIASES:
            raise InvalidInputError(f"Unsupported export format {format!r}; "
                                    "use html, pdf, md or json.")
        return ALIASES[key]
    suffix = Path(path).suffix.lower()
    if suffix not in FORMATS:
        raise InvalidInputError(
            f"Cannot infer the export format from {str(path)!r}; use a .html, .pdf, .md or "
            ".json extension or pass format=...")
    return FORMATS[suffix]


def export(result: AnalysisResult, path: str | Path, format: str | None = None,
           **kwargs: Any) -> Path:
    """Write ``result`` to ``path`` in the requested (or inferred) format."""
    fmt = infer_format(path, format)
    out = Path(path)
    if fmt == "html":
        result.to_html(out, **kwargs)
    elif fmt == "pdf":
        result.to_pdf(out, **kwargs)
    elif fmt == "markdown":
        result.to_markdown(out, **kwargs)
    else:
        kwargs.pop("include_figures", None)  # JSON never contains figures
        result.to_json(out, **kwargs)
    return out
