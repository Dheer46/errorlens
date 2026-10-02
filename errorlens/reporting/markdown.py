"""Render a :class:`Report` as Markdown (figures are written to a sidecar asset folder)."""

from __future__ import annotations

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


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(columns: list[str], rows: list[list[str]], align: list[str]) -> str:
    head = "| " + " | ".join(_cell(c) for c in columns) + " |"
    sep = "| " + " | ".join("---:" if a == "r" else "---" for a in align) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_block(block: Block, asset_dir: str | None) -> str:
    if isinstance(block, Heading):
        return "#" * min(block.level + 1, 6) + " " + block.text
    if isinstance(block, Paragraph):
        if block.style == "lead":
            return f"**{block.text}**"
        if block.style == "muted":
            return f"_{block.text}_"
        return block.text
    if isinstance(block, KeyValues):
        return _table(["Item", "Value"], [[k, v] for k, v in block.items], ["l", "l"])
    if isinstance(block, Tiles):
        return _table([t[0] for t in block.items], [[f"**{t[1]}**" for t in block.items],
                                                    [t[2] for t in block.items]],
                      ["r"] * len(block.items))
    if isinstance(block, Table):
        out = _table(block.columns, block.rows, block.alignment())
        return out + (f"\n\n_{block.caption}_" if block.caption else "")
    if isinstance(block, PatternCard):
        conds = "\n".join(("AND " if i else "") + c for i, c in enumerate(block.conditions))
        lines = [f"#### Pattern #{block.rank}", "", "```text", conds, "```", "",
                 block.headline, "", _table(["Statistic", "Value"],
                                            [[k, v] for k, v in block.stats], ["l", "l"])]
        if block.note:
            lines += ["", f"_{block.note}_"]
        return "\n".join(lines)
    if isinstance(block, Figure):
        if asset_dir is None:
            return f"_[Figure omitted: {block.caption}]_"
        return f"![{block.caption}]({asset_dir}/{block.key}.png)\n\n_{block.caption}_"
    if isinstance(block, Callout):
        prefix = "**Warning:** " if block.level == "warning" else "**Note:** "
        return "> " + prefix + block.text
    if isinstance(block, BulletList):
        return "\n".join(f"- {i}" for i in block.items)
    raise TypeError(f"Unknown block type {type(block).__name__}")


def render_markdown(report: Report, asset_dir: str | None) -> str:
    parts = [f"# {report.title}", "", f"_{report.subtitle}_", "",
             _table(["Item", "Value"], [list(m) for m in report.meta], ["l", "l"]), "",
             "## Contents", ""]
    parts += [f"- [{s.title}](#{s.anchor})" for s in report.sections]
    for s in report.sections:
        parts += ["", f'<a id="{s.anchor}"></a>', "", f"## {s.title}", ""]
        for b in s.blocks:
            parts += [render_block(b, asset_dir), ""]
    parts += ["---", f"_{report.footer}_", ""]
    return "\n".join(parts)


def write_markdown(result: AnalysisResult, path: str | Path | None = None,
                   include_figures: bool = True) -> str:
    """Render ``result`` as Markdown.

    When ``path`` is given and figures are included, PNG files are written to
    ``<stem>_assets/`` next to the Markdown file and referenced relatively. Without a path,
    figures are omitted (Markdown has no portable way to embed images).
    """
    from errorlens.reporting.report import build_report

    report = build_report(result, include_figures=include_figures and path is not None)
    asset_dir = None
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        figures = report.figures()
        if figures:
            asset_dir = f"{out.stem}_assets"
            folder = out.parent / asset_dir
            folder.mkdir(exist_ok=True)
            for fig in figures:
                (folder / f"{fig.key}.png").write_bytes(fig.png)
        text = render_markdown(report, asset_dir)
        out.write_text(text, encoding="utf-8")
        return text
    return render_markdown(report, None)
