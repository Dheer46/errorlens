"""PDF export: inspect the generated document, not just that it was written."""

import io

import pytest

pypdf = pytest.importorskip("pypdf")
pytest.importorskip("reportlab")


def _read(path):
    reader = pypdf.PdfReader(str(path))
    text = "\n".join(page.extract_text() for page in reader.pages)
    n_images = sum(len(page.images) for page in reader.pages)
    return reader, text, n_images


@pytest.fixture(scope="module")
def binary_pdf(binary_result, tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "report.pdf"
    out = binary_result.to_pdf(path)
    assert out == path
    return path


def test_pdf_is_valid_and_multipage(binary_pdf):
    data = binary_pdf.read_bytes()
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    reader, text, _ = _read(binary_pdf)
    assert len(reader.pages) > 3
    assert f"Page 1 of {len(reader.pages)}" in text
    assert f"Page {len(reader.pages)} of {len(reader.pages)}" in text
    assert reader.metadata.title == "ErrorLens Error Analysis"
    assert reader.metadata.author == "ErrorLens"


def test_pdf_contains_report_content(binary_pdf, binary_result):
    _, text, _ = _read(binary_pdf)
    for heading in ("Executive summary", "Model performance", "Failure patterns",
                    "Feature error analysis", "False-positive analysis",
                    "False-negative analysis", "Statistical analysis", "Methodology",
                    "Limitations"):
        assert heading in text, heading
    # tables: header cells and values from the pattern table
    for cell in ("Lift", "Adj. p", "95% CI", "Precision", "Recall"):
        assert cell in text
    top = binary_result.patterns[0]
    assert f"PATTERN #{top.rank}" in text
    assert str(top.conditions[0]).split(" ")[0] in text
    assert f"{top.n_samples:,}" in text
    # Unicode glyphs render through the embedded font
    assert "≤" in text or "<=" in text


def test_pdf_embeds_charts(binary_pdf, binary_result):
    _, _, n_images = _read(binary_pdf)
    from errorlens.reporting.report import build_report

    assert n_images == len(build_report(binary_result).figures()) >= 5


def test_pdf_without_figures(binary_result, tmp_path):
    path = binary_result.to_pdf(tmp_path / "plain.pdf", include_figures=False)
    _, text, n_images = _read(path)
    assert n_images == 0
    assert "Failure patterns" in text


def test_long_pattern_lists_paginate(regression_result, tmp_path):
    """Many patterns + long tables must flow across pages without layout errors."""
    from errorlens.reporting.document import Report, Section, Table
    from errorlens.reporting.pdf import render_pdf

    rows = [[str(i), "feature_" + "x" * 60 + f" <= {i} AND other_feature > {i}", f"{i:,}",
             "12.5%", "1.50x"] for i in range(300)]
    report = Report("Long table", "stress test", "now", [("Rows", "300")],
                    [Section("Patterns", "p", [Table(["#", "Pattern", "Rows", "Rate", "Lift"],
                                                     rows, "caption",
                                                     ["r", "l", "r", "r", "r"])])],
                    False, "footer")
    buf = io.BytesIO()
    render_pdf(report, buf)
    reader = pypdf.PdfReader(io.BytesIO(buf.getvalue()))
    assert len(reader.pages) >= 5
    # the header row is repeated on continuation pages
    assert all("Pattern" in page.extract_text() for page in reader.pages)
    assert "299" in reader.pages[-1].extract_text()

    path = regression_result.to_pdf(tmp_path / "regression.pdf")
    reader, text, n_images = _read(path)
    assert "Residual analysis" in text and n_images >= 3
