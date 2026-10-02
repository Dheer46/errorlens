"""Format-neutral report document model.

The report builder (:mod:`errorlens.reporting.report`) decides *what* a report contains and
expresses it with these blocks. Renderers (HTML, PDF, Markdown) only decide *how* blocks are
laid out, which guarantees that every format carries the same analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Heading:
    text: str
    level: int = 2


@dataclass
class Paragraph:
    text: str
    style: Literal["normal", "lead", "muted"] = "normal"


@dataclass
class KeyValues:
    items: list[tuple[str, str]]


@dataclass
class Tiles:
    """Headline numbers (executive summary)."""

    items: list[tuple[str, str, str]]  # (label, value, note)


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]
    caption: str = ""
    align: list[str] | None = None  # "l" or "r" per column
    widths: list[float] | None = None  # relative widths (PDF)

    def alignment(self) -> list[str]:
        return self.align or ["l"] * len(self.columns)


@dataclass
class PatternCard:
    rank: int
    conditions: list[str]
    headline: str
    stats: list[tuple[str, str]]
    note: str = ""


@dataclass
class Figure:
    key: str
    png: bytes
    caption: str
    width_in: float
    height_in: float


@dataclass
class Callout:
    text: str
    level: Literal["info", "warning"] = "info"


@dataclass
class BulletList:
    items: list[str]


Block = (Heading | Paragraph | KeyValues | Tiles | Table | PatternCard | Figure | Callout
         | BulletList)


@dataclass
class Section:
    title: str
    anchor: str
    blocks: list[Block] = field(default_factory=list)

    def add(self, *blocks: Block) -> Section:
        self.blocks.extend(blocks)
        return self


@dataclass
class Report:
    title: str
    subtitle: str
    created_at: str
    meta: list[tuple[str, str]]
    sections: list[Section]
    figures_included: bool
    footer: str

    def figures(self) -> list[Figure]:
        return [b for s in self.sections for b in s.blocks if isinstance(b, Figure)]
