"""Render a :class:`Report` as a paginated PDF using reportlab (optional dependency).

Layout features: A4 pages, title block, section headings, wrapped table cells, tables that
paginate with repeated header rows, pattern cards kept on one page, embedded charts scaled
to the frame, "Page X of Y" footers and PDF metadata.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

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
from errorlens.utils.optional import require

if TYPE_CHECKING:
    from errorlens.core.results import AnalysisResult

ACCENT = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
BORDER = "#d9d8d3"
HEADER_BG = "#f1f0ec"
INFO_BG = "#e8f1fc"
WARN_BG = "#fff4dc"
WARN = "#9a5b00"

# Fallbacks for glyphs missing from the built-in Helvetica (WinAnsi) fonts.
ASCII_FALLBACK = {"≤": "<=", "≥": ">=", "−": "-", "×": "x", "α": "alpha", "ε": "eps",
                  "→": "->", "ρ": "rho", "²": "2", "≈": "~", "…": "..."}


def _reportlab() -> Any:
    return require("reportlab", package="reportlab", extra="pdf", feature="PDF export")


class _Fonts:
    regular = "Helvetica"
    bold = "Helvetica-Bold"
    mono = "Courier"
    unicode = False


def _register_fonts() -> type[_Fonts]:
    """Use DejaVu (bundled with matplotlib) for full Unicode coverage when available."""
    try:
        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return _Fonts
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    files = {"EL-Sans": "DejaVuSans.ttf", "EL-Sans-Bold": "DejaVuSans-Bold.ttf",
             "EL-Mono": "DejaVuSansMono.ttf"}
    if not all((font_dir / f).exists() for f in files.values()):
        return _Fonts
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, file in files.items():
        if name not in registered:
            pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))

    class Unicode(_Fonts):
        regular = "EL-Sans"
        bold = "EL-Sans-Bold"
        mono = "EL-Mono"
        unicode = True

    return Unicode


class _PdfBuilder:
    def __init__(self, report: Report) -> None:
        _reportlab()
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm

        self.report = report
        self.colors = colors
        self.fonts = _register_fonts()
        self.page_size = A4
        self.margin = 17 * mm
        self.frame_width = A4[0] - 2 * self.margin
        f = self.fonts
        c = colors.HexColor
        self.styles = {
            "brand": ParagraphStyle("brand", fontName=f.bold, fontSize=9, textColor=c(ACCENT),
                                    leading=12, spaceAfter=2),
            "title": ParagraphStyle("title", fontName=f.bold, fontSize=22, leading=27,
                                    textColor=c(TEXT), spaceAfter=2),
            "subtitle": ParagraphStyle("subtitle", fontName=f.regular, fontSize=11, leading=15,
                                       textColor=c(TEXT_2), spaceAfter=10),
            "h2": ParagraphStyle("h2", fontName=f.bold, fontSize=15, leading=19,
                                 textColor=c(TEXT), spaceBefore=6, spaceAfter=4),
            "h3": ParagraphStyle("h3", fontName=f.bold, fontSize=11, leading=14,
                                 textColor=c(TEXT), spaceBefore=8, spaceAfter=3),
            "body": ParagraphStyle("body", fontName=f.regular, fontSize=9.5, leading=13.5,
                                   textColor=c(TEXT), spaceAfter=4),
            "lead": ParagraphStyle("lead", fontName=f.bold, fontSize=10, leading=14,
                                   textColor=c(TEXT), spaceBefore=6, spaceAfter=3),
            "muted": ParagraphStyle("muted", fontName=f.regular, fontSize=9, leading=12.5,
                                    textColor=c(TEXT_2), spaceAfter=4),
            "cell": ParagraphStyle("cell", fontName=f.regular, fontSize=7.8, leading=9.8,
                                   textColor=c(TEXT)),
            "cell_r": ParagraphStyle("cell_r", fontName=f.regular, fontSize=7.8, leading=9.8,
                                     textColor=c(TEXT), alignment=2),
            "head": ParagraphStyle("head", fontName=f.bold, fontSize=7.8, leading=9.8,
                                   textColor=c(TEXT_2)),
            "head_r": ParagraphStyle("head_r", fontName=f.bold, fontSize=7.8, leading=9.8,
                                     textColor=c(TEXT_2), alignment=2),
            "caption": ParagraphStyle("caption", fontName=f.regular, fontSize=7.8, leading=10,
                                      textColor=c(MUTED), spaceBefore=2, spaceAfter=8),
            "kv_key": ParagraphStyle("kv_key", fontName=f.regular, fontSize=8.6, leading=11,
                                     textColor=c(TEXT_2)),
            "kv_val": ParagraphStyle("kv_val", fontName=f.regular, fontSize=8.6, leading=11,
                                     textColor=c(TEXT)),
            "tile_label": ParagraphStyle("tile_label", fontName=f.bold, fontSize=7.2,
                                         leading=9, textColor=c(TEXT_2)),
            "tile_value": ParagraphStyle("tile_value", fontName=f.bold, fontSize=17,
                                         leading=21, textColor=c(TEXT)),
            "tile_note": ParagraphStyle("tile_note", fontName=f.regular, fontSize=7.5,
                                        leading=9.5, textColor=c(MUTED)),
            "card_rank": ParagraphStyle("card_rank", fontName=f.bold, fontSize=8,
                                        leading=10, textColor=c(ACCENT)),
            "card_cond": ParagraphStyle("card_cond", fontName=f.mono, fontSize=9,
                                        leading=12.5, textColor=c(TEXT)),
            "bullet": ParagraphStyle("bullet", fontName=f.regular, fontSize=9.3, leading=13,
                                     textColor=c(TEXT)),
        }
        self.mm = mm

    # ------------------------------------------------------------------ text helpers
    def text(self, value: str) -> str:
        if not self.fonts.unicode:
            for src, dst in ASCII_FALLBACK.items():
                value = value.replace(src, dst)
        return escape(value)

    def para(self, value: str, style: str) -> Any:
        from reportlab.platypus import Paragraph as RLParagraph

        return RLParagraph(self.text(value), self.styles[style])

    # ------------------------------------------------------------------ blocks
    def kv_table(self, items: list[tuple[str, str]], width: float | None = None) -> Any:
        from reportlab.platypus import Table as RLTable
        from reportlab.platypus import TableStyle

        width = width or self.frame_width
        data = [[self.para(k, "kv_key"), self.para(v, "kv_val")] for k, v in items]
        t = RLTable(data, colWidths=[width * 0.38, width * 0.62], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, self.colors.HexColor(BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return t

    def kv_grid(self, items: list[tuple[str, str]], width: float) -> Any:
        """Two label/value pairs per row — compact statistics block for pattern cards."""
        from reportlab.platypus import Table as RLTable
        from reportlab.platypus import TableStyle

        half = (len(items) + 1) // 2
        left, right = items[:half], items[half:]
        data = []
        for i in range(half):
            row = [self.para(left[i][0], "kv_key"), self.para(left[i][1], "kv_val")]
            if i < len(right):
                row += [self.para(right[i][0], "kv_key"), self.para(right[i][1], "kv_val")]
            else:
                row += ["", ""]
            data.append(row)
        t = RLTable(data, colWidths=[width * 0.21, width * 0.29, width * 0.21, width * 0.29],
                    hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, self.colors.HexColor(BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (2, 0), (2, -1), 10),
        ]))
        return t

    def data_table(self, block: Table) -> list[Any]:
        from reportlab.platypus import LongTable, TableStyle

        align = block.alignment()
        widths = block.widths or [1.0] * len(block.columns)
        total = sum(widths)
        col_widths = [self.frame_width * w / total for w in widths]
        header = [self.para(c, "head_r" if a == "r" else "head")
                  for c, a in zip(block.columns, align)]
        rows = [[self.para(v, "cell_r" if a == "r" else "cell") for v, a in zip(row, align)]
                for row in block.rows]
        t = LongTable([header, *rows], colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.colors.HexColor(HEADER_BG)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.35, self.colors.HexColor(BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        out = [t]
        if block.caption:
            out.append(self.para(block.caption, "caption"))
        else:
            from reportlab.platypus import Spacer

            out.append(Spacer(1, 6))
        return out

    def tiles(self, block: Tiles) -> Any:
        from reportlab.platypus import Table as RLTable
        from reportlab.platypus import TableStyle

        n = len(block.items)
        cells = [[self.para(lbl.upper(), "tile_label"), self.para(val, "tile_value"),
                  self.para(note, "tile_note")] for lbl, val, note in block.items]
        gap = 4 * self.mm
        width = (self.frame_width - gap * (n - 1)) / n
        inner = []
        for c in cells:
            box = RLTable([[c[0]], [c[1]], [c[2]]], colWidths=[width])
            box.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, self.colors.HexColor(BORDER)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
                ("TOPPADDING", (0, 1), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 1),
            ]))
            inner.append(box)
        row: list[Any] = []
        widths: list[float] = []
        for i, box in enumerate(inner):
            row.append(box)
            widths.append(width)
            if i < n - 1:
                row.append("")
                widths.append(gap)
        t = RLTable([row], colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        return t

    def card(self, block: PatternCard) -> Any:
        from reportlab.platypus import KeepTogether, TableStyle
        from reportlab.platypus import Table as RLTable

        inner_w = self.frame_width - 12
        conds = "<br/>".join(
            (f"<font color='{MUTED}'>AND</font> " if i else "") + self.text(c)
            for i, c in enumerate(block.conditions))
        from reportlab.platypus import Paragraph as RLParagraph

        rows: list[list[Any]] = [
            [self.para(f"PATTERN #{block.rank}", "card_rank")],
            [RLParagraph(conds, self.styles["card_cond"])],
            [self.para(block.headline, "body")],
            [self.kv_grid(block.stats, inner_w - 18)],
        ]
        if block.note:
            rows.append([self.para(block.note, "muted")])
        t = RLTable(rows, colWidths=[inner_w], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, self.colors.HexColor(BORDER)),
            ("LINEBEFORE", (0, 0), (0, -1), 3, self.colors.HexColor(ACCENT)),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        from reportlab.platypus import Spacer

        return KeepTogether([t, Spacer(1, 8)])

    def figure(self, block: Figure) -> Any:
        from reportlab.platypus import Image, KeepTogether

        max_w = self.frame_width
        max_h = (self.page_size[1] - 2 * self.margin) * 0.62
        w = max_w
        h = w * block.height_in / block.width_in
        if h > max_h:
            h = max_h
            w = h * block.width_in / block.height_in
        img = Image(io.BytesIO(block.png), width=w, height=h)
        img.hAlign = "CENTER"
        return KeepTogether([img, self.para(block.caption, "caption")])

    def callout(self, block: Callout) -> Any:
        from reportlab.platypus import Table as RLTable
        from reportlab.platypus import TableStyle

        prefix = "Warning: " if block.level == "warning" else ""
        t = RLTable([[self.para(prefix + block.text, "body")]], colWidths=[self.frame_width])
        style: list[tuple[Any, ...]] = [
            ("BACKGROUND", (0, 0), (-1, -1),
             self.colors.HexColor(WARN_BG if block.level == "warning" else INFO_BG)),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if block.level == "warning":
            style.append(("LINEBEFORE", (0, 0), (0, -1), 2.5, self.colors.HexColor(WARN)))
        t.setStyle(TableStyle(style))
        return t

    def bullets(self, block: BulletList) -> Any:
        from reportlab.platypus import ListFlowable, ListItem

        return ListFlowable(
            [ListItem(self.para(i, "bullet"), leftIndent=12, value="•") for i in block.items],
            bulletType="bullet", start="•", leftIndent=12, bulletFontName=self.fonts.regular,
            bulletFontSize=8)

    def block(self, block: Block) -> list[Any]:
        from reportlab.platypus import Spacer

        if isinstance(block, Heading):
            return [self.para(block.text, "h3")]
        if isinstance(block, Paragraph):
            return [self.para(block.text, {"lead": "lead", "muted": "muted"}.get(block.style,
                                                                                 "body"))]
        if isinstance(block, KeyValues):
            return [self.kv_table(block.items), Spacer(1, 6)]
        if isinstance(block, Tiles):
            return [self.tiles(block), Spacer(1, 6)]
        if isinstance(block, Table):
            return self.data_table(block)
        if isinstance(block, PatternCard):
            return [self.card(block)]
        if isinstance(block, Figure):
            return [self.figure(block)]
        if isinstance(block, Callout):
            return [self.callout(block), Spacer(1, 4)]
        if isinstance(block, BulletList):
            return [self.bullets(block), Spacer(1, 4)]
        raise TypeError(f"Unknown block type {type(block).__name__}")

    # ------------------------------------------------------------------ document
    def story(self) -> list[Any]:
        from reportlab.platypus import CondPageBreak, HRFlowable, Spacer

        r = self.report
        story: list[Any] = [
            self.para("ERRORLENS", "brand"),
            self.para(r.title, "title"),
            self.para(r.subtitle, "subtitle"),
            self.kv_table(r.meta, self.frame_width * 0.75),
            Spacer(1, 10),
        ]
        for section in r.sections:
            story.append(CondPageBreak(60 * self.mm))
            story.append(self.para(section.title, "h2"))
            story.append(HRFlowable(width="100%", thickness=1.2,
                                    color=self.colors.HexColor(ACCENT), spaceBefore=0,
                                    spaceAfter=7))
            for b in section.blocks:
                story.extend(self.block(b))
            story.append(Spacer(1, 10))
        return story

    def build(self, target: str | Path | io.BytesIO) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.platypus import SimpleDocTemplate

        fonts = self.fonts
        footer = self.text(self.report.footer)
        title = self.text(self.report.title)
        margin = self.margin
        colors = self.colors

        class NumberedCanvas(rl_canvas.Canvas):
            """Two-pass canvas so each page can show 'Page X of Y'."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._saved: list[dict[str, Any]] = []

            def showPage(self) -> None:
                self._saved.append(dict(self.__dict__))
                self._startPage()

            def save(self) -> None:
                total = len(self._saved)
                for state in self._saved:
                    self.__dict__.update(state)
                    self._decorate(total)
                    super().showPage()
                super().save()

            def _decorate(self, total: int) -> None:
                width, height = A4
                self.saveState()
                self.setStrokeColor(colors.HexColor(BORDER))
                self.setLineWidth(0.5)
                self.line(margin, margin - 7, width - margin, margin - 7)
                self.setFont(fonts.regular, 7.5)
                self.setFillColor(colors.HexColor(MUTED))
                self.drawString(margin, margin - 17, f"{footer}  ·  {title}"
                                if fonts.unicode else f"{footer} - {title}")
                self.drawRightString(width - margin, margin - 17,
                                     f"Page {self._pageNumber} of {total}")
                if self._pageNumber > 1:
                    self.drawString(margin, height - margin + 8, "ErrorLens report")
                self.restoreState()

        doc = SimpleDocTemplate(
            target if isinstance(target, io.BytesIO) else str(target),
            pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=margin,
            bottomMargin=margin + 6, title=self.report.title, author="ErrorLens",
            subject="ML model error analysis", creator=self.report.footer,
            keywords="ErrorLens, error analysis, failure patterns")
        doc.build(self.story(), canvasmaker=NumberedCanvas)


def render_pdf(report: Report, path: str | Path | io.BytesIO) -> None:
    """Render a report document to ``path`` (or a BytesIO buffer)."""
    _PdfBuilder(report).build(path)


def write_pdf(result: AnalysisResult, path: str | Path, include_figures: bool = True) -> Path:
    """Render ``result`` to a PDF file at ``path``.

    Raises:
        MissingDependencyError: if reportlab is not installed (``pip install errorlens[pdf]``).
    """
    _reportlab()  # fail fast with installation instructions before doing any work
    from errorlens.reporting.report import build_report

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(build_report(result, include_figures=include_figures), out)
    return out
