"""Render a :class:`Report` as a single self-contained HTML file (no external assets)."""

from __future__ import annotations

import base64
from html import escape
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from errorlens.reporting.document import (
    Block,
    BulletList,
    Callout,
    Figure,
    Heading,
    KeyValues,
    Paragraph,
    PatternCard,
    Report,
    Table,
    Tiles,
)

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult


def _css() -> str:
    return resources.files("errorlens.reporting").joinpath("templates/report.css").read_text(
        encoding="utf-8")


def _kv(items: list[tuple[str, str]]) -> str:
    rows = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in items)
    return f'<dl class="kv">{rows}</dl>'


def render_block(block: Block) -> str:
    if isinstance(block, Heading):
        level = min(max(block.level, 3), 4)
        return f"<h{level}>{escape(block.text)}</h{level}>"
    if isinstance(block, Paragraph):
        cls = f' class="{block.style}"' if block.style != "normal" else ""
        return f"<p{cls}>{escape(block.text)}</p>"
    if isinstance(block, KeyValues):
        return _kv(block.items)
    if isinstance(block, Tiles):
        tiles = "".join(
            f'<div class="tile"><div class="label">{escape(lbl)}</div>'
            f'<div class="value">{escape(val)}</div><div class="note">{escape(note)}</div></div>'
            for lbl, val, note in block.items)
        return f'<div class="tiles">{tiles}</div>'
    if isinstance(block, Table):
        align = block.alignment()
        head = "".join(f'<th class="{a}">{escape(c)}</th>' for c, a in zip(block.columns, align))
        body = "".join(
            "<tr>" + "".join(f'<td class="{a}">{escape(v)}</td>' for v, a in zip(row, align))
            + "</tr>" for row in block.rows)
        caption = f"<caption>{escape(block.caption)}</caption>" if block.caption else ""
        return (f'<div class="table-wrap"><table>{caption}<thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>")
    if isinstance(block, PatternCard):
        conds = "<br>".join(
            (f'<span class="and">AND</span> {escape(c)}' if i else escape(c))
            for i, c in enumerate(block.conditions))
        note = f'<div class="note">{escape(block.note)}</div>' if block.note else ""
        return (f'<div class="card"><div class="rank">Pattern #{block.rank}</div>'
                f'<div class="conditions">{conds}</div>'
                f'<div class="headline">{escape(block.headline)}</div>'
                f"{_kv(block.stats)}{note}</div>")
    if isinstance(block, Figure):
        data = base64.b64encode(block.png).decode("ascii")
        return (f'<figure><img alt="{escape(block.caption)}" src="data:image/png;base64,{data}">'
                f"<figcaption>{escape(block.caption)}</figcaption></figure>")
    if isinstance(block, Callout):
        return f'<div class="callout {block.level}">{escape(block.text)}</div>'
    if isinstance(block, BulletList):
        items = "".join(f"<li>{escape(i)}</li>" for i in block.items)
        return f"<ul>{items}</ul>"
    raise TypeError(f"Unknown block type {type(block).__name__}")


def render_html(report: Report) -> str:
    meta = "".join(f"<span>{escape(k)}: <b>{escape(v)}</b></span>" for k, v in report.meta)
    toc = "".join(f'<a href="#{s.anchor}">{escape(s.title)}</a>' for s in report.sections)
    sections = "".join(
        f'<section class="report-section" id="{s.anchor}"><h2>{escape(s.title)}</h2>'
        + "".join(render_block(b) for b in s.blocks) + "</section>"
        for s in report.sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="{escape(report.footer)}">
<title>{escape(report.title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
<header class="report-header">
<div class="brand">ErrorLens</div>
<h1>{escape(report.title)}</h1>
<p class="subtitle">{escape(report.subtitle)}</p>
<div class="meta">{meta}</div>
</header>
<nav class="toc" aria-label="Sections">{toc}</nav>
{sections}
<footer>{escape(report.footer)}</footer>
</div>
</body>
</html>
"""


def write_html(result: AnalysisResult, path: str | Path | None = None,
               include_figures: bool = True) -> str:
    """Render ``result`` as HTML; write it to ``path`` if given and return the HTML."""
    from errorlens.reporting.report import build_report

    html = render_html(build_report(result, include_figures=include_figures))
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html
