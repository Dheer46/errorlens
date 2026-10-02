import json
import sys
from html.parser import HTMLParser

import pytest

from errorlens import InvalidInputError, MissingDependencyError
from errorlens.reporting.document import Figure, PatternCard, Table
from errorlens.reporting.export import infer_format
from errorlens.reporting.report import build_report


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: dict[str, int] = {}
        self.ids: set[str] = set()
        self.imgs: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "img":
            self.imgs.append(attrs.get("src", ""))


REQUIRED_SECTIONS = ["summary", "dataset", "performance", "errors", "patterns", "features",
                     "statistics", "warnings", "methodology", "limitations"]


def test_report_document_shared_by_all_formats(binary_result):
    report = build_report(binary_result)
    anchors = [s.anchor for s in report.sections]
    for a in [*REQUIRED_SECTIONS, "false-positives", "false-negatives"]:
        assert a in anchors
    assert report.figures_included
    blocks = [b for s in report.sections for b in s.blocks]
    assert any(isinstance(b, PatternCard) for b in blocks)
    assert any(isinstance(b, Table) for b in blocks)
    assert any(isinstance(b, Figure) for b in blocks)


def test_html_report(binary_result, tmp_path):
    path = tmp_path / "report.html"
    html = binary_result.to_html(path)
    assert path.exists() and path.read_text(encoding="utf-8") == html
    parser = _Collector()
    parser.feed(html)
    for a in REQUIRED_SECTIONS:
        assert a in parser.ids
    assert parser.tags["table"] >= 4
    assert parser.imgs and all(src.startswith("data:image/png;base64,") for src in parser.imgs)
    top = binary_result.patterns[0]
    assert top.description.replace("<", "&lt;").split(" AND ")[0] in html
    assert "not mean its features cause" in html
    assert "<script" not in html  # fully static


def test_html_without_figures(binary_result):
    html = binary_result.to_html(include_figures=False)
    assert "<img" not in html


def test_regression_html(regression_result, tmp_path):
    html = regression_result.to_html(tmp_path / "r.html")
    assert 'id="residuals"' in html and "High-error regions" in html


def test_markdown_report(binary_result, tmp_path):
    path = tmp_path / "out" / "report.md"
    text = binary_result.to_markdown(path)
    assert path.exists()
    assets = tmp_path / "out" / "report_assets"
    pngs = list(assets.glob("*.png"))
    assert pngs and all(p.read_bytes()[:4] == b"\x89PNG" for p in pngs)
    assert "](report_assets/" in text
    assert "| # | Pattern |" in text
    assert "## Methodology" in text and "## Limitations" in text
    # string output without a path omits figures
    assert "report_assets" not in binary_result.to_markdown()


def test_json_report(binary_result, regression_result, tmp_path):
    path = tmp_path / "report.json"
    text = binary_result.to_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == json.loads(text)
    assert data["metadata"]["task"] == "classification"
    assert len(data["patterns"]) == len(binary_result.patterns)
    p0 = data["patterns"][0]
    for key in ("conditions", "n_samples", "n_errors", "rate", "baseline", "lift", "coverage",
                "p_value", "p_value_adjusted", "complexity", "stats", "validation"):
        assert key in p0
    assert p0["validation"]["ci_low"] <= p0["validation"]["value"] <= p0["validation"]["ci_high"]
    assert set(data["directional_patterns"]) == {"false_positive", "false_negative"}
    assert data["discovery_statistics"][0]["family_size"] >= 1
    assert "Multiple-testing correction" in data["methodology"]
    reg = json.loads(regression_result.to_json())
    assert reg["residual_analysis"]["interpretation"]
    assert "NaN" not in text and "Infinity" not in text


def test_export_dispatch(binary_result, tmp_path):
    for ext in ("html", "md", "json"):
        out = binary_result.export(tmp_path / f"r.{ext}")
        assert out.exists() and out.stat().st_size > 500
    out = binary_result.export(tmp_path / "report.txt", format="json")
    json.loads(out.read_text(encoding="utf-8"))
    with pytest.raises(InvalidInputError):
        binary_result.export(tmp_path / "report.docx")
    with pytest.raises(InvalidInputError):
        binary_result.export(tmp_path / "report.html", format="xlsx")
    assert infer_format("a.MARKDOWN") == "markdown"


def test_pdf_missing_dependency_error(binary_result, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "reportlab", None)
    with pytest.raises(MissingDependencyError) as info:
        binary_result.to_pdf(tmp_path / "x.pdf")
    msg = str(info.value)
    assert 'pip install "errorlens[pdf]"' in msg and "reportlab" in msg
    assert isinstance(info.value, ImportError)
    assert not (tmp_path / "x.pdf").exists()


def test_missing_matplotlib_degrades_reports_but_plot_raises(binary_result, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setattr(binary_result, "_figure_cache", None)
    html = binary_result.to_html()
    assert "<img" not in html and "matplotlib is not installed" in html
    with pytest.raises(MissingDependencyError, match="errorlens\\[viz\\]"):
        binary_result.plot_patterns()


def test_plots_return_figures(binary_result, regression_result, multiclass_result):
    import matplotlib.pyplot as plt

    figs = [binary_result.plot(), binary_result.plot_patterns(),
            binary_result.plot_patterns(kind="false_negative"),
            binary_result.plot_feature_errors(), binary_result.plot_feature_errors("region"),
            binary_result.plot_error_distribution(),
            binary_result.plot_error_distribution("region"),
            binary_result.plot_confusion_matrix(normalize=True),
            binary_result.plot_error_heatmap(), binary_result.plot_error_heatmap("age", "region"),
            regression_result.plot(), regression_result.plot_residuals(),
            regression_result.plot_error_distribution(), multiclass_result.plot()]
    for fig in figs:
        assert fig.axes
    plt.close("all")
    with pytest.raises(InvalidInputError):
        binary_result.plot_patterns(kind="nope")
    with pytest.raises(InvalidInputError):
        binary_result.plot_error_heatmap("age")
